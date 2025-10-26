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
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_so101_follower import SO101FollowerConfig

logger = logging.getLogger(__name__)


class SO101Follower(Robot):
    """
    SO-101 Follower Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = SO101FollowerConfig
    name = "so101_follower"

    def __init__(self, config: SO101FollowerConfig):
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
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            if not self.calibration:
                logger.info("No calibration file found, running calibration")
            else:
                logger.info("Mismatch between calibration values in the motor and the calibration file")
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # self.calibration is not empty here
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

        if 'gripper' in range_mins:
            # add negative offset to gripper to enable just-tight follower grip when leader gripper is closed
            gripper_adjust_offset_deg = 4
            encoding_table = self.bus.model_encoding_table.get(self.bus.motors['gripper'].model, {})
            homing_offset_bits = encoding_table.get("Homing_Offset")
            full_range = 1 << (homing_offset_bits + 1)
            gripper_adjust_offset = - (int)(full_range * gripper_adjust_offset_deg / 360)
            original_min = range_mins['gripper']
            adjusted_min = original_min + gripper_adjust_offset
            print(f"Gripper range adjusted: original min={original_min} -> adjusted min={adjusted_min} (offset={gripper_adjust_offset})")
            range_mins['gripper'] = adjusted_min

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
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.bus.write("P_Coefficient", motor, 16)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 0)

                if motor == "gripper":
                    self.bus.write(
                        "Max_Torque_Limit", motor, 500
                    )  # 50% of the max torque limit to avoid burnout
                    self.bus.write("Protection_Current", motor, 250)  # 50% of max current to avoid burnout
                    self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded

    def setup_motors(self) -> None:
        expected_ids = [1]
        # Check if there are other motors on the bus
        succ, msg = self._check_unexpected_motors_on_bus(expected_ids=expected_ids, raise_on_error=True)
        if not succ:
            input(msg)
            succ, msg = self._check_unexpected_motors_on_bus(expected_ids=expected_ids, raise_on_error=True)

        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor ONLY and press enter.")
            succ, msg = self._check_unexpected_motors_on_bus(expected_ids=expected_ids, raise_on_error=False)
            if not succ:
                input(msg)
                succ, msg = self._check_unexpected_motors_on_bus(expected_ids=expected_ids, raise_on_error=False)            
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")
            expected_ids.append(self.bus.motors[motor].id)

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

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
