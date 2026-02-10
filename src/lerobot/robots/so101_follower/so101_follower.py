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
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

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
            gripper_adjust_offset_deg = 3.5
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
        # # Check if there are other motors on the bus
        # succ, msg = self._check_unexpected_motors_on_bus(expected_ids=expected_ids, raise_on_error=True)
        # if not succ:
        #     input(msg)
        #     succ, msg = self._check_unexpected_motors_on_bus(expected_ids=expected_ids, raise_on_error=True)

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
    
    def _force_modify_motor_id(self, current_id: int, model_number: int, target_id: int) -> None:
        """
        Force modify a motor's ID from current_id to target_id.
        
        Args:
            current_id: Current motor ID
            model_number: Motor model number
            target_id: Target motor ID (should be 1 in setup loop)
        """
        # Find model name from model_number
        from lerobot.motors.feetech.tables import MODEL_NUMBER_TABLE
        model_name = None
        for model, num in MODEL_NUMBER_TABLE.items():
            if num == model_number:
                model_name = model
                break
        
        if model_name is None:
            raise RuntimeError(f"Unknown model number: {model_number}")
        
        # Get current baudrate
        current_baudrate = self.bus.get_baudrate()
        
        # Disable torque
        self.bus._disable_torque(current_id, model_name)
        
        # Write new ID
        from lerobot.motors.motors_bus import get_address
        addr, length = get_address(self.bus.model_ctrl_table, model_name, "ID")
        self.bus._write(addr, length, current_id, target_id)
        
        # Restore baudrate
        self.bus.set_baudrate(current_baudrate)
    
    def _check_unexpected_motors_on_bus(self, expected_ids: list[int], raise_on_error: bool = True, target_motor_id: int | None = None, in_setup_loop: bool = False) -> None:
        """
        Check if there are other motors on the bus, if there are other motors, stop the setup process.
        
        Args:
            expected_ids: List of motor IDs that are expected to be on the bus
            raise_on_error: If True, raise RuntimeError on error; otherwise return (False, message)
            target_motor_id: The target motor ID to set (used in force modify prompt). If None, uses expected_ids[0] if available.
            in_setup_loop: If True, we're in the setup loop. Check if there's exactly 1 motor with target_motor_id. 
                          If ID doesn't match, ask user to force modify to target_motor_id.
        
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
            # If in_setup_loop is True, we're in the setup loop and check if motor ID matches target_motor_id
            if in_setup_loop:
                if len(found_motors) != 1:
                    if raise_on_error:
                        raise RuntimeError(
                            f"Expected exactly 1 motor on the bus, but found {len(found_motors)} motors: {list(found_motors.keys())}"
                        )
                    else:
                        return False, f"Expected exactly 1 motor on the bus, but found {len(found_motors)} motors. Please connect only the motor to be set up and press ENTER to try again."
                
                # There is exactly 1 motor, check if its ID matches the target ID
                found_motor_id = list(found_motors.keys())[0]
                found_model_number = found_motors[found_motor_id]
                # target_motor_id should be provided when in_setup_loop is True
                if target_motor_id is None:
                    target_motor_id = expected_ids[0] if expected_ids else 1
                
                if found_motor_id == target_motor_id:
                    # Motor ID matches target, OK to proceed with setup
                    return True, "OK"
                else:
                    # Motor ID doesn't match target, ask user if they want to force modify to target ID
                    user_input = input(
                        f"There is only 1 motor on the bus with ID {found_motor_id}, but it's not motor ID {target_motor_id}. "
                        f"Do you want to force modify its ID to {target_motor_id}? (yes/no): "
                    )
                    if user_input.strip().lower() == "yes":
                        # Force modify the motor ID to target ID
                        self._force_modify_motor_id(found_motor_id, found_model_number, target_id=target_motor_id)
                        logger.info(f"Motor ID has been modified from {found_motor_id} to {target_motor_id}.")
                        return True, "OK"
                    else:
                        # User declined, treat as error
                        if raise_on_error:
                            raise RuntimeError(
                                f"Motor ID modification cancelled. Found motor ID {found_motor_id}, expected motor ID {target_motor_id}."
                            )
                        else:
                            logger.warning(
                                f"Motor ID modification cancelled. Found motor ID {found_motor_id}, expected motor ID {target_motor_id}."
                            )
                            return False, "Please unplug the motor and press ENTER to try again."
            
            # If target_motor_id is provided (not in setup loop), check if there is exactly 1 motor
            # and verify its ID matches the target
            elif target_motor_id is not None:
                if len(found_motors) != 1:
                    if raise_on_error:
                        raise RuntimeError(
                            f"Expected exactly 1 motor on the bus, but found {len(found_motors)} motors: {list(found_motors.keys())}"
                        )
                    else:
                        return False, f"Expected exactly 1 motor on the bus, but found {len(found_motors)} motors. Please connect only the motor to be set up and press ENTER to try again."
                
                # There is exactly 1 motor, check its ID
                found_motor_id = list(found_motors.keys())[0]
                if found_motor_id != target_motor_id:
                    # Motor ID doesn't match, ask user if they want to force modify
                    user_input = input(
                        f"There is only 1 motor on the bus with ID {found_motor_id}, but it's not motor ID {target_motor_id}. "
                        f"Do you want to force modify its ID to {target_motor_id}? (yes/no): "
                    )
                    if user_input.strip().lower() == "yes":
                        logger.info(f"User confirmed to force modify motor ID from {found_motor_id} to {target_motor_id}.")
                        return True, "OK"
                    else:
                        # User declined, treat as error
                        if raise_on_error:
                            raise RuntimeError(
                                f"Motor ID modification cancelled. Found motor ID {found_motor_id}, expected motor ID {target_motor_id}."
                            )
                        else:
                            logger.warning(
                                f"Motor ID modification cancelled. Found motor ID {found_motor_id}, expected motor ID {target_motor_id}."
                            )
                            return False, "Please unplug the motor and press ENTER to try again."
                else:
                    # Motor ID matches, OK
                    return True, "OK"
            
            # Normal check: Check if there are other motors on the bus
            unexpected_motors = [motor_id for motor_id in found_motors.keys() if motor_id not in expected_ids]
            
            if unexpected_motors:
                unexpected_motors_str = ", ".join(map(str, sorted(unexpected_motors)))
                # Special case: if there is only 1 motor on the bus and it's not in expected_ids,
                # ask user if they want to force modify the motor ID
                if len(found_motors) == 1 and len(unexpected_motors) == 1:
                    motor_id = unexpected_motors[0]
                    # Determine target ID: use first expected_id
                    target_id = expected_ids[0] if expected_ids else 1
                    user_input = input(
                        f"There is only 1 motor on the bus with ID {motor_id}, but it's not motor ID {target_id}. "
                        f"Do you want to force modify its ID to {target_id}? (yes/no): "
                    )
                    if user_input.strip().lower() == "yes":
                        logger.info(f"User confirmed to force modify motor ID from {motor_id} to {target_id}.")
                        return True, "OK"
                    else:
                        # User declined, treat as error
                        if raise_on_error:
                            raise RuntimeError(
                                f"Motor ID modification cancelled. Found motor ID {motor_id}, expected motor ID {target_id}."
                            )
                        else:
                            logger.warning(
                                f"Motor ID modification cancelled. Found motor ID {motor_id}, expected motor ID {target_id}."
                            )
                            return False, "Please unplug the motor and press ENTER to try again."
                
                # Normal case: multiple unexpected motors or not the special case above
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
