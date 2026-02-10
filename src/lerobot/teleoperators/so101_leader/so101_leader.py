#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_so101_leader import SO101LeaderConfig

logger = logging.getLogger(__name__)


class SO101Leader(Teleoperator):
    """
    SO-101 Leader Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = SO101LeaderConfig
    name = "so101_leader"

    def __init__(self, config: SO101LeaderConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            if not self.calibration:
                logger.info("No calibration file found, running calibration")
            else:
                logger.info("Mismatch between calibration values in the motor and the calibration file")
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        expected_ids = [1]

        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor ONLY and press ENTER.")
            target_motor_id = self.bus.motors[motor].id  # 目标序号i
            
            # Check motor on bus according to the rules
            should_setup = self._check_and_confirm_motor_setup(target_motor_id)
            if should_setup:
                self.bus.setup_motor(motor)
                print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")
                expected_ids.append(self.bus.motors[motor].id)
            else:
                # Skipped because motor ID already matches target
                print(f"'{motor}' motor ID is already {target_motor_id}, skipping setup.")
                expected_ids.append(target_motor_id)

    def _check_and_confirm_motor_setup(self, target_motor_id: int) -> bool:
        """
        Check motor on bus and confirm if setup is needed according to the rules:
        - Rule a: At most 1 motor on bus, raise error if more than 1
        - Rule b: If only 1 motor with ID 1 (factory default), return True to setup
        - Rule c: If motor ID already matches target_motor_id, return False to skip
        - Rule d: If motor ID is neither 1 nor target_motor_id, ask user confirmation
        
        Args:
            target_motor_id: Target motor ID (i) for current joint
            
        Returns:
            True if setup should proceed, False if should skip
        """
        # Ensure the bus is connected
        if not self.bus.is_connected:
            self.bus.connect(handshake=False)
        
        # Scan all motors at the current baudrate
        current_baudrate = self.bus.get_baudrate()
        self.bus.set_baudrate(current_baudrate)
        
        # Scan all motors on the bus
        found_motors = self.bus.broadcast_ping(raise_on_error=False)
        
        if found_motors is None:
            # If the scan fails, try other baudrates
            for baudrate in self.bus.available_baudrates:
                if baudrate == current_baudrate:
                    continue
                    
                self.bus.set_baudrate(baudrate)
                found_motors = self.bus.broadcast_ping(raise_on_error=False)
                if found_motors is not None:
                    break
        
        # Restore the original baudrate
        self.bus.set_baudrate(current_baudrate)
        
        if found_motors is None:
            raise RuntimeError("No motors found on the bus. Please connect the motor and try again.")
        
        # Rule a: At most 1 motor on bus
        if len(found_motors) > 1:
            motor_ids = list(found_motors.keys())
            raise RuntimeError(
                f"Expected at most 1 motor on the bus, but found {len(found_motors)} motors with IDs: {motor_ids}. "
                f"Please connect only the motor to be set up."
            )
        
        if len(found_motors) == 0:
            raise RuntimeError("No motors found on the bus. Please connect the motor and try again.")
        
        # Only 1 motor found
        found_motor_id = list(found_motors.keys())[0]
        
        # Rule b: If motor ID is 1 (factory default), proceed with setup
        if found_motor_id == 1:
            return True
        
        # Rule c: If motor ID already matches target, skip setup
        if found_motor_id == target_motor_id:
            return False
        
        # Rule d: Motor ID is neither 1 nor target_motor_id, ask user confirmation
        user_input = input(
            f"There is 1 motor on the bus with ID {found_motor_id}, not factory default ID 1. "
            f"Do you want to force modify its ID to {target_motor_id}? (yes/no): "
        )
        if user_input.strip().lower() == "yes":
            # User confirmed, proceed with setup (setup_motor will handle the ID modification)
            return True
        else:
            # User declined, raise error
            raise RuntimeError(
                f"Motor ID modification cancelled. Found motor ID {found_motor_id}, expected motor ID {target_motor_id}."
            )
    
    def _check_unexpected_motors_on_bus(self, expected_ids: list[int], raise_on_error: bool = True) -> None:
        """
        Check if there are other motors on the bus, if there are other motors, stop the setup process.        
        Raises:
            RuntimeError: If there are other motors on the bus, stop the setup process.
        """
        # Ensure the bus is connected
        if not self.bus.is_connected:
            self.bus.connect(handshake=False)
        
        # Scan all motors at the current baudrate
        current_baudrate = self.bus.get_baudrate()
        self.bus.set_baudrate(current_baudrate)
        
        # Scan all motors on the bus
        found_motors = self.bus.broadcast_ping(raise_on_error=False)
        
        if found_motors is None:
            # If the scan fails, try other baudrates
            for baudrate in self.bus.available_baudrates:
                if baudrate == current_baudrate:
                    continue
                    
                self.bus.set_baudrate(baudrate)
                found_motors = self.bus.broadcast_ping(raise_on_error=False)
                if found_motors is not None:
                    break
        
        # Restore the original baudrate
        self.bus.set_baudrate(current_baudrate)
        
        if found_motors is not None:
            # Check if there are other motors on the bus
            unexpected_motors = [motor_id for motor_id in found_motors.keys() if motor_id not in expected_ids]
            
            if unexpected_motors:
                unexpected_motors_str = ", ".join(map(str, sorted(unexpected_motors)))
                if raise_on_error:
                    raise RuntimeError(
                        f"There are unexpected motors on the bus: {unexpected_motors_str}. "
                        f"Seems this arm has been setup before, not necessary to setup again."
                    )
                else:
                    logger.warning(
                        f"There are unexpected motors on the bus: {unexpected_motors_str}. "
                    )
                    return False, "Please unplug the last motor and press ENTER to try again."
            return True, "OK"
        
        return False, "No motors found on the bus, please connect the arm and press ENTER to try again."

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
