"""
Remote teleoperation for XLerobot using:
- one local `BiSOLeader` as the dual-arm master
- one local keyboard for the mobile base
- one remote `XLerobotClient` as the dual-arm follower + base

Run robot host on the robot side first:

```bash
PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot
```

Then run this teleop script on the operator side.
"""

from __future__ import annotations

import time

import numpy as np

from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient, XLerobotClientConfig
from lerobot.teleoperators.bi_so101_leader import BiSO101Leader, BiSO101LeaderConfig
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so101_leader import SO101LeaderConfig
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


ARM_ACTION_MAP = {
    "left_shoulder_pan.pos": "left_arm_shoulder_pan.pos",
    "left_shoulder_lift.pos": "left_arm_shoulder_lift.pos",
    "left_elbow_flex.pos": "left_arm_elbow_flex.pos",
    "left_wrist_flex.pos": "left_arm_wrist_flex.pos",
    "left_wrist_roll.pos": "left_arm_wrist_roll.pos",
    "left_gripper.pos": "left_arm_gripper.pos",
    "right_shoulder_pan.pos": "right_arm_shoulder_pan.pos",
    "right_shoulder_lift.pos": "right_arm_shoulder_lift.pos",
    "right_elbow_flex.pos": "right_arm_elbow_flex.pos",
    "right_wrist_flex.pos": "right_arm_wrist_flex.pos",
    "right_wrist_roll.pos": "right_arm_wrist_roll.pos",
    "right_gripper.pos": "right_arm_gripper.pos",
}


def map_bi_leader_action_to_xlerobot(raw_action: dict[str, float]) -> dict[str, float]:
    return {target_key: raw_action[source_key] for source_key, target_key in ARM_ACTION_MAP.items() if source_key in raw_action}


def main():
    fps = 50

    remote_ip = "192.168.200.104"
    robot_name = "joyandai_xlerobot1"

    left_leader_port = "COM68"
    right_leader_port = "COM69"
    leader_id = "my_bi_so101_leader"

    robot = XLerobotClient(XLerobotClientConfig(remote_ip=remote_ip, id=robot_name))
    leader = BiSO101Leader(
        BiSO101LeaderConfig(
            id=leader_id,
            left_arm_config=SO101LeaderConfig(port=left_leader_port, id=f"{leader_id}_left"),
            right_arm_config=SO101LeaderConfig(port=right_leader_port, id=f"{leader_id}_right"),
        )
    )
    keyboard = KeyboardTeleop(KeyboardTeleopConfig(id="my_laptop_keyboard"))

    try:
        robot.connect()
        leader.connect()
        keyboard.connect()

        if not robot.is_connected or not leader.is_connected or not keyboard.is_connected:
            raise ValueError("Robot, bi leader, or keyboard is not connected.")

        init_rerun(session_name="xlerobot_remote_bi_so101_teleop")
        print("Starting remote teleop with bi_so101 leader for arms and keyboard for base...")
        print("Base keys: i/k/j/l move, u/o rotate, n/m speed +/-")

        while True:
            loop_start = time.perf_counter()

            leader_action = map_bi_leader_action_to_xlerobot(leader.get_action())
            pressed_keys = np.array(list(keyboard.get_action().keys()))
            base_action = robot._from_keyboard_to_base_action(pressed_keys)

            action = {**leader_action, **base_action}
            sent_action = robot.send_action(action)

            obs = robot.get_observation()
            log_rerun_data(obs, sent_action)

            pressed_key_set = set(pressed_keys.tolist())
            if robot.teleop_keys["quit"] in pressed_key_set:
                print("Quit key pressed, exiting teleop loop.")
                break

            dt = time.perf_counter() - loop_start
            remaining = max(1.0 / fps - dt, 0.0)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        if keyboard.is_connected:
            keyboard.disconnect()
        if leader.is_connected:
            leader.disconnect()
        if robot.is_connected:
            robot.disconnect()
        print("Teleoperation ended.")


if __name__ == "__main__":
    main()
