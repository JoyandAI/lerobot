#!/usr/bin/env python3

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

"""Auto calibration example with CLI arguments."""

import argparse
import logging

from lerobot.motors.auto_calibrate import AutoCalibrateConfig, auto_calibrate_connected_device
from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa: F401
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config, so_leader  # noqa: F401
from lerobot.teleoperators.so_leader import SO101LeaderConfig
from lerobot.utils.utils import init_logging

DEVICE_CONFIG_FACTORIES = {
    "tele": SO101LeaderConfig,
    "robot": SO101FollowerConfig,
}

DEVICE_FACTORIES = {
    "tele": make_teleoperator_from_config,
    "robot": make_robot_from_config,
}

AUTO_CALIBRATION_DEFAULTS = {
    "tele": {
        "try_torque": 400,
        "max_torque": 500,
        "torque_step": 50,
        "explore_velocity": 600,
        "wait_time_s": 0.5,
        "velocity_threshold": 4,
        "position_tolerance": 4000,
    },
    "robot": {
        "try_torque": 600,
        "max_torque": 1000,
        "torque_step": 50,
        "explore_velocity": 800,
        "wait_time_s": 0.5,
        "velocity_threshold": 4,
        "position_tolerance": 4000,
    },
}

AUTO_CALIBRATION_OVERRIDE_FIELDS = (
    "try_torque",
    "max_torque",
    "torque_step",
    "explore_velocity",
    "wait_time_s",
    "velocity_threshold",
    "position_tolerance",
)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run auto calibration for a SO101 teleoperator or robot.")
    parser.add_argument("--port", required=True, help="Serial port, for example COM9.")
    parser.add_argument(
        "--device-type",
        required=True,
        choices=["tele", "robot"],
        help="Choose whether to calibrate a teleoperator or a robot.",
    )
    parser.add_argument("--id", default="my_so101", help="Device id used for calibration file naming.")
    parser.add_argument(
        "--calibration-dir",
        default=None,
        help="Optional calibration output directory. Defaults to the project/device standard path.",
    )
    parser.add_argument("--try-torque", type=int, default=None, help="Override try_torque.")
    parser.add_argument("--max-torque", type=int, default=None, help="Override max_torque.")
    parser.add_argument("--torque-step", type=int, default=None, help="Override torque_step.")
    parser.add_argument("--explore-velocity", type=int, default=None, help="Override explore_velocity.")
    parser.add_argument("--wait-time-s", type=float, default=None, help="Override wait_time_s.")
    parser.add_argument("--velocity-threshold", type=int, default=None, help="Override velocity_threshold.")
    parser.add_argument("--position-tolerance", type=int, default=None, help="Override position_tolerance.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation before calibration starts.",
    )
    parser.add_argument(
        "--log",
        type=parse_bool,
        dest="log_enabled",
        default=True,
        help="Whether to print logger output in terminal. Default is true. Example: --log=false",
    )
    return parser


def build_device_config(args: argparse.Namespace) -> RobotConfig | TeleoperatorConfig:
    config_kwargs = {
        "port": args.port,
        "id": args.id,
    }
    if args.calibration_dir is not None:
        config_kwargs["calibration_dir"] = args.calibration_dir

    return DEVICE_CONFIG_FACTORIES[args.device_type](**config_kwargs)


def build_auto_calibrate_config(
    device_config: RobotConfig | TeleoperatorConfig,
    args: argparse.Namespace,
) -> AutoCalibrateConfig:
    config_values = dict(AUTO_CALIBRATION_DEFAULTS[args.device_type])
    for field_name in AUTO_CALIBRATION_OVERRIDE_FIELDS:
        override_value = getattr(args, field_name)
        if override_value is not None:
            config_values[field_name] = override_value

    return AutoCalibrateConfig(robot=device_config, **config_values)


def build_device(args: argparse.Namespace, device_config: RobotConfig | TeleoperatorConfig):
    return DEVICE_FACTORIES[args.device_type](device_config)


def print_calibration_summary(calibration_dict):
    print("\nCalibration results:")
    print("=" * 60)
    for motor_name, calib in calibration_dict.items():
        print(f"\nMotor: {motor_name}")
        print(f"  ID: {calib.id}")
        print(f"  Homing Offset: {calib.homing_offset}")
        print(f"  Range Min: {calib.range_min}")
        print(f"  Range Max: {calib.range_max}")
    print("=" * 60)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.log_enabled:
        init_logging()
    else:
        logging.disable(logging.CRITICAL)

    device_config = build_device_config(args)
    auto_calib_config = build_auto_calibrate_config(device_config, args)
    device = build_device(args, device_config)

    print("Connecting device...")
    device.connect(calibrate=False)

    try:
        print("\nStarting auto calibration...")
        print("Warning: make sure the robot workspace is clear before continuing.")
        if not args.yes:
            input("Press ENTER to continue...")

        result = auto_calibrate_connected_device(device, auto_calib_config)

        print("\nAuto calibration completed.")
        if result.calibration_path is not None:
            print(f"Calibration file saved to: {result.calibration_path}")

        print_calibration_summary(result.calibration_dict)
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user.")
    except Exception as error:
        print(f"\nCalibration failed: {error}")
        raise
    finally:
        print("\nDisconnecting device...")
        device.disconnect()


if __name__ == "__main__":
    main()
