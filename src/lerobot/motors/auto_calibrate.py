#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Auto calibration helpers for Feetech-based devices."""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from pprint import pformat
from threading import Event
from typing import Protocol

import draccus

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots import Robot, RobotConfig, make_robot_from_config, so_follower  # noqa: F401
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)
_CALIBRATION_PAUSED = Event()

SERVO_RESOLUTION = 4096
DEFAULT_CALIBRATION_ORDER = (
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
    "shoulder_pan",
    "shoulder_lift",
)


class Direction(Enum):
    """Direction used when exploring a mechanical limit."""

    CLOCKWISE = "clockwise"
    ANTI_CLOCKWISE = "anti_clockwise"


class CalibratableDevice(Protocol):
    """Minimal device interface required by the calibration workflow."""

    bus: FeetechMotorsBus
    calibration_fpath: Path | str
    name: str


@dataclass(frozen=True)
class MotorAction:
    motor_name: str
    direction: Direction


@dataclass(frozen=True)
class JointCalibrationBehavior:
    first_direction: Direction = Direction.CLOCKWISE
    second_direction: Direction = Direction.ANTI_CLOCKWISE
    midpoint_velocity_sign: int = 1
    midpoint_half_offset_adjustment: int = 0
    use_closed_position_as_center: bool = False
    stop_midpoint_motion_immediately: bool = False


@dataclass(frozen=True)
class LimitExplorationResult:
    first_offset: int
    first_limit: int
    second_offset: int
    second_limit: int


@dataclass(frozen=True)
class RobotCalibrationPlan:
    ordered_motors: tuple[str, ...]
    recovery_order: tuple[str, ...]
    pre_actions: dict[str, tuple[MotorAction, ...]] = field(default_factory=dict)
    post_actions: dict[str, tuple[MotorAction, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class AutoCalibrateResult:
    calibration_dict: dict[str, MotorCalibration]
    calibration_path: Path | None = None


PRE_CALIBRATION_ACTIONS = {
    "shoulder_lift": (
        MotorAction("wrist_roll", Direction.ANTI_CLOCKWISE),
        MotorAction("elbow_flex", Direction.ANTI_CLOCKWISE),
    ),
}

POST_CALIBRATION_ACTIONS = {
    "shoulder_lift": (MotorAction("shoulder_lift", Direction.ANTI_CLOCKWISE),),
    "elbow_flex": (MotorAction("elbow_flex", Direction.CLOCKWISE),),
    "wrist_flex": (MotorAction("wrist_flex", Direction.CLOCKWISE),),
}


@dataclass
class AutoCalibrateConfig:
    """Auto calibration configuration."""

    robot: RobotConfig
    try_torque: int = 400
    max_torque: int = 600
    wrist_roll_torque_limit: int = 300
    gripper_torque_limit: int = 300
    shoulder_pan_midpoint_adjustment: int = -150
    torque_step: int = 50
    explore_velocity: int = 600
    wait_time_s: float = 0.3
    velocity_threshold: int = 4
    position_tolerance: int = 4000

    OVER_LOAD_BIT = 0x20


def set_calibration_paused(paused: bool):
    """Pause or resume the current calibration workflow."""

    if paused:
        _CALIBRATION_PAUSED.set()
    else:
        _CALIBRATION_PAUSED.clear()


def is_calibration_paused() -> bool:
    """Return whether calibration is currently paused."""

    return _CALIBRATION_PAUSED.is_set()


def wait_if_calibration_paused(
    bus: FeetechMotorsBus | None = None,
    motor_name: str | None = None,
    resume_velocity: int | None = None,
):
    """Block while calibration is paused and safely stop motor motion if needed."""

    if not is_calibration_paused():
        return

    if bus is not None and motor_name is not None and resume_velocity is not None:
        bus.write("Goal_Velocity", motor_name, 0, normalize=False)

    logger.info("Calibration paused.")
    while is_calibration_paused():
        time.sleep(0.1)
    logger.info("Calibration resumed.")

    if bus is not None and motor_name is not None and resume_velocity is not None:
        bus.write("Goal_Velocity", motor_name, resume_velocity, normalize=False)


def sleep_with_pause(
    duration_s: float,
    bus: FeetechMotorsBus | None = None,
    motor_name: str | None = None,
    resume_velocity: int | None = None,
):
    """Sleep in small intervals so calibration can be paused responsively."""

    remaining = max(duration_s, 0.0)
    while remaining > 0:
        wait_if_calibration_paused(bus, motor_name, resume_velocity)
        chunk = min(0.05, remaining)
        time.sleep(chunk)
        remaining -= chunk


def normalize_homing_offset(offset: int, bits: int = 11) -> int:
    """Fold homing offset into the signed range accepted by Feetech."""

    max_offset = (1 << bits) - 1
    period = 1 << (bits + 1)

    while offset > max_offset:
        offset -= period
    while offset < -max_offset:
        offset += period

    return offset


def fold_unwrapped_range_to_single_turn(
    range_min: int,
    range_max: int,
    offset: int,
    max_position: int,
) -> tuple[int, int, int]:
    """Shift an unwrapped range back into a single-turn window."""

    resolution = max_position + 1

    while range_min < 0:
        range_min += resolution
        range_max += resolution
        offset -= resolution

    while range_max > max_position:
        range_min -= resolution
        range_max -= resolution
        offset += resolution

    return range_min, range_max, offset


def unwrap_physical_limit_range(
    first_limit: int,
    second_limit: int,
    second_direction: Direction,
    resolution: int,
) -> tuple[int, int]:
    """Represent the two physical limits as one increasing unwrapped range."""

    if second_direction == Direction.CLOCKWISE:
        return first_limit, first_limit + ((second_limit - first_limit + resolution) % resolution)

    return second_limit, second_limit + ((first_limit - second_limit + resolution) % resolution)


def compute_directional_offset(
    previous_position: int,
    current_position: int,
    direction: Direction,
    resolution: int,
) -> int:
    """Compute the wrapped offset traveled in the requested direction."""

    if direction == Direction.CLOCKWISE:
        return (current_position - previous_position + resolution) % resolution
    return (previous_position - current_position + resolution) % resolution


def get_joint_behavior(motor_name: str) -> JointCalibrationBehavior:
    """Collect the special-case behavior for a motor without changing its rules."""

    is_shoulder_lift = "shoulder_lift" in motor_name
    is_shoulder_pan = "shoulder_pan" in motor_name
    is_wrist_roll = "wrist_roll" in motor_name
    is_gripper = "gripper" in motor_name.lower()

    behavior = JointCalibrationBehavior()
    if is_shoulder_lift:
        behavior = JointCalibrationBehavior(
            first_direction=Direction.ANTI_CLOCKWISE,
            second_direction=Direction.CLOCKWISE,
            midpoint_velocity_sign=-1,
            stop_midpoint_motion_immediately=True,
        )

    if is_wrist_roll:
        behavior = JointCalibrationBehavior(
            first_direction=behavior.first_direction,
            second_direction=behavior.second_direction,
            midpoint_velocity_sign=behavior.midpoint_velocity_sign,
            midpoint_half_offset_adjustment=behavior.midpoint_half_offset_adjustment,
            use_closed_position_as_center=behavior.use_closed_position_as_center,
            stop_midpoint_motion_immediately=True,
        )

    if is_shoulder_pan:
        behavior = JointCalibrationBehavior(
            first_direction=behavior.first_direction,
            second_direction=behavior.second_direction,
            midpoint_velocity_sign=behavior.midpoint_velocity_sign,
            midpoint_half_offset_adjustment=behavior.midpoint_half_offset_adjustment,
            use_closed_position_as_center=behavior.use_closed_position_as_center,
            stop_midpoint_motion_immediately=behavior.stop_midpoint_motion_immediately,
        )

    if is_gripper:
        behavior = JointCalibrationBehavior(
            first_direction=behavior.first_direction,
            second_direction=behavior.second_direction,
            midpoint_velocity_sign=behavior.midpoint_velocity_sign,
            midpoint_half_offset_adjustment=behavior.midpoint_half_offset_adjustment,
            use_closed_position_as_center=True,
            stop_midpoint_motion_immediately=behavior.stop_midpoint_motion_immediately,
        )

    return behavior


def order_motors_for_calibration(
    motor_names: list[str],
    preferred_order: tuple[str, ...] = DEFAULT_CALIBRATION_ORDER,
) -> list[str]:
    """Preserve the current calibration order through an explicit rule list."""

    ordered: list[str] = []
    for joint_name in preferred_order:
        ordered.extend(name for name in motor_names if joint_name in name and name not in ordered)

    ordered.extend(name for name in motor_names if name not in ordered)
    return ordered


def build_robot_calibration_plan(
    robot: CalibratableDevice,
    motor_names: list[str],
) -> RobotCalibrationPlan:
    """Build the current robot-level calibration plan from explicit rules."""

    motors_to_calibrate = list(motor_names)
    if getattr(robot, "name", None) == "xlehead":
        motors_to_calibrate = [motor for motor in motors_to_calibrate if motor.startswith("head")]

    return RobotCalibrationPlan(
        ordered_motors=tuple(order_motors_for_calibration(motors_to_calibrate)),
        recovery_order=tuple(motors_to_calibrate),
        pre_actions=PRE_CALIBRATION_ACTIONS,
        post_actions=POST_CALIBRATION_ACTIONS,
    )


def run_motor_actions(
    bus: FeetechMotorsBus,
    actions: tuple[MotorAction, ...],
    config: AutoCalibrateConfig,
):
    """Run a fixed sequence of exploration actions."""

    for action in actions:
        explore_literal_limit(bus, action.motor_name, action.direction, config)


def run_matching_actions(
    bus: FeetechMotorsBus,
    motor_name: str,
    action_map: dict[str, tuple[MotorAction, ...]],
    config: AutoCalibrateConfig,
):
    """Run every action whose trigger matches the current motor name."""

    for trigger, actions in action_map.items():
        if trigger in motor_name:
            run_motor_actions(bus, actions, config)


def stop_motor_motion(bus: FeetechMotorsBus, motor_name: str):
    """Best-effort stop used around calibration phase boundaries."""

    try:
        bus.write("Goal_Velocity", motor_name, 0, normalize=False)
        if "wrist_roll" in motor_name:
            bus.write("Torque_Enable", motor_name, 0, normalize=False)
    except Exception:
        logger.exception("Failed to stop motor velocity for %s", motor_name)


def get_motor_torque_limit(motor_name: str, config: AutoCalibrateConfig) -> int:
    if "wrist_roll" in motor_name:
        return config.wrist_roll_torque_limit
    if "gripper" in motor_name:
        return config.gripper_torque_limit
    return config.try_torque


def get_motor_max_torque_limit(motor_name: str, config: AutoCalibrateConfig) -> int:
    if "wrist_roll" in motor_name:
        return config.wrist_roll_torque_limit
    if "gripper" in motor_name:
        return config.gripper_torque_limit
    return config.max_torque


def explore_literal_limit(
    bus: FeetechMotorsBus,
    motor_name: str,
    direction: Direction,
    config: AutoCalibrateConfig,
) -> tuple[int, int]:
    """Explore a single mechanical limit and return traveled offset and stop position."""

    logger.info(f"Exploring {motor_name} {direction.value} limit...")

    resolution = SERVO_RESOLUTION
    current_torque = get_motor_torque_limit(motor_name, config)
    motor_max_torque = get_motor_max_torque_limit(motor_name, config)
    previous_position = bus.read("Present_Position", motor_name, normalize=False)
    start_position = previous_position
    limit_position = previous_position
    still_count = 0

    goal_velocity = config.explore_velocity if direction == Direction.CLOCKWISE else -config.explore_velocity

    bus.write("Operating_Mode", motor_name, OperatingMode.VELOCITY.value, normalize=False)
    bus.write("Torque_Limit", motor_name, current_torque, normalize=False)
    bus.write("Torque_Enable", motor_name, 1, normalize=False)
    bus.write("Goal_Velocity", motor_name, goal_velocity, normalize=False)

    while True:
        sleep_with_pause(config.wait_time_s, bus, motor_name, goal_velocity)

        try:
            current_position = bus.read("Present_Position", motor_name, normalize=False)
        except RuntimeError as error:
            if "Overload error" in str(error):
                logger.info("Motor overloaded, releasing torque before reading current position again.")
                while True:
                    try:
                        bus.write("Torque_Limit", motor_name, 0, normalize=False)
                        sleep_with_pause(0.2, bus, motor_name, 0)
                    except RuntimeError:
                        continue
                    break

                current_position = bus.read("Present_Position", motor_name, normalize=False)
                compute_directional_offset(previous_position, current_position, direction, resolution)
                limit_position = current_position
                break
            raise

        compute_directional_offset(previous_position, current_position, direction, resolution)
        limit_position = current_position

        current_velocity = bus.read("Present_Velocity", motor_name, normalize=False)
        if abs(current_velocity) <= config.velocity_threshold and current_position == previous_position:
            still_count += 1
            logger.info(
                "Motor appears stationary: "
                f"pos={current_position}, velocity={current_velocity}, still_count={still_count}"
            )

            if still_count >= 1:
                logger.info(f"Velocity near zero and position stable. Limit found at {current_position}.")
                break

            if current_torque < motor_max_torque:
                current_torque = min(current_torque + config.torque_step, motor_max_torque)
                bus.write("Torque_Limit", motor_name, current_torque, normalize=False)
        else:
            still_count = 0

        status = bus.read("Status", motor_name, normalize=False)
        if status & config.OVER_LOAD_BIT != 0:
            logger.info("Motor status indicates overload. Stopping exploration.")
            break

        previous_position = current_position

    total_offset = compute_directional_offset(start_position, current_position, direction, resolution)

    stop_motor_motion(bus, motor_name)
    return total_offset, limit_position


def log_limit_result(step_index: int, direction: Direction, limit_position: int, traveled_offset: int):
    """Log one limit exploration result."""

    logger.info(f"\nStep {step_index}: explore {direction.value} limit...")
    logger.info(f"{direction.value} limit position: {limit_position}")
    logger.info(f"{direction.value} traveled offset: {traveled_offset}")


def run_limit_exploration_sequence(
    bus: FeetechMotorsBus,
    motor_name: str,
    config: AutoCalibrateConfig,
    behavior: JointCalibrationBehavior,
) -> LimitExplorationResult:
    """Run the two-step limit search for one joint."""

    first_offset, first_limit = explore_literal_limit(bus, motor_name, behavior.first_direction, config)
    log_limit_result(1, behavior.first_direction, first_limit, first_offset)

    second_offset, second_limit = explore_literal_limit(bus, motor_name, behavior.second_direction, config)
    log_limit_result(2, behavior.second_direction, second_limit, second_offset)

    return LimitExplorationResult(
        first_offset=first_offset,
        first_limit=first_limit,
        second_offset=second_offset,
        second_limit=second_limit,
    )


def move_motor_to_midpoint(
    bus: FeetechMotorsBus,
    motor_name: str,
    config: AutoCalibrateConfig,
    behavior: JointCalibrationBehavior,
    half_offset: int,
) -> int:
    """Replay the current midpoint motion logic without changing stop conditions."""

    current_position = bus.read("Present_Position", motor_name, normalize=False)
    midpoint_position = (current_position + half_offset) % SERVO_RESOLUTION

    logger.info(f"Step 3: half offset: {half_offset}")
    logger.info(f"Step 3: computed midpoint: {midpoint_position}")
    logger.info("Step 4: moving toward midpoint...")

    bus.write("Operating_Mode", motor_name, OperatingMode.VELOCITY.value, normalize=False)
    bus.write("Torque_Limit", motor_name, get_motor_torque_limit(motor_name, config), normalize=False)
    bus.write("Torque_Enable", motor_name, 1, normalize=False)
    bus.write(
        "Goal_Velocity",
        motor_name,
        behavior.midpoint_velocity_sign * config.explore_velocity,
        normalize=False,
    )

    previous_position = bus.read("Present_Position", motor_name, normalize=False)
    moved_offset = 0
    while True:
        wait_if_calibration_paused(
            bus,
            motor_name,
            behavior.midpoint_velocity_sign * config.explore_velocity,
        )

        if behavior.stop_midpoint_motion_immediately:
            stop_motor_motion(bus, motor_name)
            break

        current_position = bus.read("Present_Position", motor_name, normalize=False)
        if moved_offset >= half_offset:
            stop_motor_motion(bus, motor_name)
            break

        delta = (current_position - previous_position + SERVO_RESOLUTION) % SERVO_RESOLUTION
        if delta > config.position_tolerance:
            previous_position = current_position
            continue

        moved_offset += delta
        previous_position = current_position
    stop_motor_motion(bus, motor_name)
    actual_midpoint = bus.read("Present_Position", motor_name, normalize=False)
    logger.info(f"Actual midpoint position: logical={actual_midpoint}, physical={actual_midpoint}")
    return actual_midpoint


def auto_calibrate_single_joint(
    bus: FeetechMotorsBus,
    motor_name: str,
    config: AutoCalibrateConfig,
) -> MotorCalibration:
    """Auto calibrate a single joint."""

    logger.info(f"\nStarting auto calibration for motor {motor_name}")
    logger.info("=" * 60)

    motor = bus.motors[motor_name]
    model = motor.model
    max_position = bus.model_resolution_table[model] - 1
    motor_try_torque = get_motor_torque_limit(motor_name, config)
    motor_max_torque = get_motor_max_torque_limit(motor_name, config)

    bus.write("Max_Torque_Limit", motor_name, motor_max_torque, normalize=False)
    bus.write("Overload_Torque", motor_name, int(motor_try_torque * 95 / motor_max_torque), normalize=False)
    bus.write("Min_Position_Limit", motor_name, 0, normalize=False)
    bus.write("Max_Position_Limit", motor_name, max_position, normalize=False)

    original_offset = bus.read("Homing_Offset", motor_name, normalize=False)
    original_present = bus.read("Present_Position", motor_name, normalize=False)
    original_physical = original_present + original_offset
    logger.info(
        f"Current position: logical={original_present}, physical={original_physical}, offset={original_offset}"
    )

    bus.write("Homing_Offset", motor_name, 0, normalize=False)
    current_present = bus.read("Present_Position", motor_name, normalize=False)
    logger.info(f"After resetting offset: logical={current_present}, physical={current_present}, offset=0")

    behavior = get_joint_behavior(motor_name)
    if "shoulder_pan" in motor_name:
        behavior = JointCalibrationBehavior(
            first_direction=behavior.first_direction,
            second_direction=behavior.second_direction,
            midpoint_velocity_sign=behavior.midpoint_velocity_sign,
            midpoint_half_offset_adjustment=config.shoulder_pan_midpoint_adjustment,
            use_closed_position_as_center=behavior.use_closed_position_as_center,
            stop_midpoint_motion_immediately=behavior.stop_midpoint_motion_immediately,
        )
    limit_result = run_limit_exploration_sequence(bus, motor_name, config, behavior)
    stop_motor_motion(bus, motor_name)

    use_physical_limit_range = motor.id in {1, 2, 3, 4, 5, 6}
    if use_physical_limit_range:
        physical_range_min, physical_range_max = unwrap_physical_limit_range(
            limit_result.first_limit,
            limit_result.second_limit,
            behavior.second_direction,
            SERVO_RESOLUTION,
        )
        total_offset = physical_range_max - physical_range_min
        logger.info(
            "Using physical limit range: "
            f"motor_id={motor.id}, motor_name={motor_name}, "
            f"first_limit={limit_result.first_limit}, second_limit={limit_result.second_limit}, "
            f"unwrapped_range=[{physical_range_min}, {physical_range_max}]"
        )
    else:
        total_offset = limit_result.second_offset

    half_offset = total_offset // 2
    half_offset += behavior.midpoint_half_offset_adjustment

    logger.info(f"\nStep 3: total offset: {total_offset}")
    if behavior.use_closed_position_as_center:
        logger.info("Step 3: gripper maps physical open/closed limits directly and skips midpoint motion.")
        actual_mid_physical = limit_result.first_limit
        logger.info(
            f"Gripper closed position: logical={limit_result.first_limit}, physical={actual_mid_physical}"
        )
    else:
        move_motor_to_midpoint(bus, motor_name, config, behavior, half_offset)
    stop_motor_motion(bus, motor_name)

    target_center = max_position // 2

    if behavior.use_closed_position_as_center:
        raw_middle_offset = physical_range_min
        ideal_offset = normalize_homing_offset(raw_middle_offset)
        ideal_range_min = 0
        ideal_range_max = total_offset
        logger.info(
            "Step 5: gripper logical range maps open to 0 and closed to 100: "
            f"open_physical={physical_range_min}, closed_physical={physical_range_max}"
        )
    elif use_physical_limit_range:
        physical_mid_unwrapped = physical_range_min + half_offset
        actual_mid_physical = physical_mid_unwrapped % SERVO_RESOLUTION
        raw_middle_offset = physical_mid_unwrapped - target_center
        ideal_offset = normalize_homing_offset(raw_middle_offset)
        ideal_range_min = target_center - half_offset
        ideal_range_max = target_center + (total_offset - half_offset)
    else:
        physical_mid_unwrapped = limit_result.second_limit + half_offset
        actual_mid_physical = physical_mid_unwrapped % SERVO_RESOLUTION
        raw_middle_offset = physical_mid_unwrapped - target_center
        ideal_offset = normalize_homing_offset(raw_middle_offset)
        ideal_range_min = target_center - half_offset
        ideal_range_max = target_center + (total_offset - half_offset)

    ideal_range_min, ideal_range_max, ideal_offset = fold_unwrapped_range_to_single_turn(
        ideal_range_min,
        ideal_range_max,
        ideal_offset,
        max_position,
    )
    ideal_offset = normalize_homing_offset(ideal_offset)

    if ideal_range_min < 0 or ideal_range_max > max_position:
        raise ValueError(
            "Calculated position limits fall outside the valid servo range: "
            f"range_min={ideal_range_min}, range_max={ideal_range_max}, max_position={max_position}"
        )

    logger.info(f"Step 5: raw midpoint offset = {raw_middle_offset}")
    logger.info(f"Step 5: final homing offset = {ideal_offset}")
    logger.info(f"Step 6: logical range = [{ideal_range_min}, {ideal_range_max}]")

    calibration = MotorCalibration(
        id=motor.id,
        drive_mode=0,
        homing_offset=ideal_offset,
        range_min=ideal_range_min,
        range_max=ideal_range_max,
    )

    bus.write("Torque_Limit", motor_name, 1000, normalize=False)
    bus.write("Protective_Torque", motor_name, 20, normalize=False)
    bus.write("Protection_Time", motor_name, 200, normalize=False)
    bus.write("Overload_Torque", motor_name, 80, normalize=False)
    stop_motor_motion(bus, motor_name)

    logger.info("\nCalibration complete.")
    logger.info(f"  Motor ID: {motor.id}")
    logger.info(f"  Homing Offset: {ideal_offset}")
    logger.info(f"  Range Min: {ideal_range_min}")
    logger.info(f"  Range Max: {ideal_range_max}")
    logger.info("=" * 60)

    return calibration


def auto_calibrate_robot(
    robot: CalibratableDevice,
    config: AutoCalibrateConfig,
) -> dict[str, MotorCalibration]:
    """Auto calibrate every motor on a connected device."""

    if not hasattr(robot, "bus") or not isinstance(robot.bus, FeetechMotorsBus):
        raise ValueError("Auto calibration only supports devices backed by FeetechMotorsBus.")

    bus = robot.bus
    if not bus.is_connected:
        raise RuntimeError("Motor bus is not connected. Connect the device before calibrating.")

    plan = build_robot_calibration_plan(robot, list(bus.motors.keys()))
    calibration_dict: dict[str, MotorCalibration] = {}

    logger.info(f"Motors selected for calibration: {list(plan.recovery_order)}")
    logger.info(f"Calibration order: {list(plan.ordered_motors)}")

    for motor_name in plan.ordered_motors:
        try:
            wait_if_calibration_paused()
            run_matching_actions(bus, motor_name, plan.pre_actions, config)
            calibration = auto_calibrate_single_joint(bus, motor_name, config)
            calibration_dict[motor_name] = calibration
            sleep_with_pause(1.0)
        except Exception as error:
            logger.error(f"Failed while calibrating motor {motor_name}: {error}")
            raise

    for motor_name in plan.recovery_order:
        try:
            wait_if_calibration_paused()
            run_matching_actions(bus, motor_name, plan.post_actions, config)
        except Exception as error:
            logger.error(f"Failed while recovering motor {motor_name}: {error}")
            raise

    logger.info("All joints recovered. Releasing torque before writing calibration registers.")
    bus.disable_torque(list(plan.recovery_order))
    bus.write_calibration(calibration_dict)

    return calibration_dict


def auto_calibrate_connected_device(
    device: CalibratableDevice,
    config: AutoCalibrateConfig,
    *,
    save: bool = True,
    calibration_path: Path | str | None = None,
) -> AutoCalibrateResult:
    """Run calibration on an already connected device and optionally save the result."""

    set_calibration_paused(False)
    calibration_dict = auto_calibrate_robot(device, config)
    saved_path: Path | None = None

    if save:
        saved_path = Path(calibration_path) if calibration_path is not None else Path(device.calibration_fpath)
        save_calibration_to_file(calibration_dict, saved_path)

    return AutoCalibrateResult(calibration_dict=calibration_dict, calibration_path=saved_path)


def save_calibration_to_file(
    calibration_dict: dict[str, MotorCalibration],
    filepath: Path | str,
):
    """Persist calibration data to a JSON file."""

    calibration_data = {}
    for motor_name, calib in calibration_dict.items():
        calibration_data[motor_name] = {
            "id": calib.id,
            "drive_mode": calib.drive_mode,
            "homing_offset": calib.homing_offset,
            "range_min": calib.range_min,
            "range_max": calib.range_max,
        }

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(calibration_data, file, indent=2)

    logger.info(f"Calibration data saved to: {filepath}")


@draccus.wrap()
def auto_calibrate(cfg: AutoCalibrateConfig):
    """CLI entry point for robot auto calibration."""

    init_logging()
    logger.info("Starting auto calibration.")
    logger.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    robot.connect(calibrate=False)

    try:
        result = auto_calibrate_connected_device(robot, cfg)
        logger.info("\nAuto calibration finished successfully.")
        if result.calibration_path is not None:
            logger.info(f"Calibration file saved to: {result.calibration_path}")
    except Exception as error:
        logger.error(f"Auto calibration failed: {error}")
        raise
    finally:
        robot.disconnect()


def main():
    auto_calibrate()


if __name__ == "__main__":
    main()
