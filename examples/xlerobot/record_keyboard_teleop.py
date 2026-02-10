# !/usr/bin/env python

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

"""
Record datasets for xlerobot robot.

This script uses keyboard control to move xlerobot robot joints while recording data.
The control method is the same as in 4_xlerobot_teleop_keyboard.py.

Example:

```shell
python -m examples.xlerobot.record \
    --robot_id=my_xlerobot \
    --remote_ip=192.168.1.123 \
    --dataset.repo_id=<hf_username>/<dataset_repo_id> \
    --dataset.num_episodes=2 \
    --dataset.single_task="My task description"
```
"""

import argparse
import logging
import time
import platform
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.model.SO101Robot import SO101Kinematics
from lerobot.processor import make_default_processors
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)

# Configuration parameters
NUM_EPISODES = 20
FPS = 30
EPISODE_TIME_SEC = 300
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "My task description"

# Keyboard mapping (same as 4_xlerobot_teleop_keyboard.py)
LEFT_KEYMAP = {
    'shoulder_pan+': 'e', 'shoulder_pan-': 'q',
    'wrist_roll+': 'r', 'wrist_roll-': 'f',
    'gripper+': 't', 'gripper-': 'g',
    'x+': 'w', 'x-': 's', 'y+': 'a', 'y-': 'd',
    'pitch+': 'z', 'pitch-': 'x',
    'reset': 'c',
    # For head motors
    "head_motor_1+": "<", "head_motor_1-": ">",
    "head_motor_2+": ",", "head_motor_2-": ".",
    
    'triangle': 'y',  # Rectangle trajectory key
}
RIGHT_KEYMAP = {
    'shoulder_pan+': '9', 'shoulder_pan-': '7',
    'wrist_roll+': '/', 'wrist_roll-': '*',
    'gripper+': '+', 'gripper-': '-',
    'x+': '8', 'x-': '2', 'y+': '4', 'y-': '6',
    'pitch+': '1', 'pitch-': '3',
    'reset': '0',

    'triangle': 'Y',  # Rectangle trajectory key
}

LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}
RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}

# Head motor mapping
HEAD_MOTOR_MAP = {
    "head_motor_1": "head_motor_1",
    "head_motor_2": "head_motor_2",
}


class SimpleHeadControl:
    """Head control class (same as 4_xlerobot_teleop_keyboard.py)"""
    def __init__(self, initial_obs, kp=1):
        self.kp = kp
        self.degree_step = 1
        # Initialize head motor positions
        self.target_positions = {
            "head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
            "head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
        }
        self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}

    def handle_keys(self, key_state):
        if key_state.get('head_motor_1+'):
            self.target_positions["head_motor_1"] += self.degree_step
        if key_state.get('head_motor_1-'):
            self.target_positions["head_motor_1"] -= self.degree_step
        if key_state.get('head_motor_2+'):
            self.target_positions["head_motor_2"] += self.degree_step
        if key_state.get('head_motor_2-'):
            self.target_positions["head_motor_2"] -= self.degree_step

    def p_control_action(self, robot):
        obs = robot.get_observation()
        action = {}
        for motor in self.target_positions:
            current = obs.get(f"{HEAD_MOTOR_MAP[motor]}.pos", 0.0)
            error = self.target_positions[motor] - current
            control = self.kp * error
            action[f"{HEAD_MOTOR_MAP[motor]}.pos"] = current + control
        return action


class SimpleTeleopArm:
    """Arm control class (same as 4_xlerobot_teleop_keyboard.py)"""
    def __init__(self, kinematics, joint_map, initial_obs, prefix="left", kp=1):
        self.kinematics = kinematics
        self.joint_map = joint_map
        self.prefix = prefix
        self.kp = kp
        # Initial joint positions
        self.joint_positions = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }
        # Set initial x/y to fixed values
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.5                    #default 0.0
        # Set the degree step and xy step
        self.degree_step = 3                # default 1
        self.xy_step = 0.006                # default 0.0021
        # Set target positions to zero for P control
        self.target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }

    def handle_keys(self, key_state):
        # Joint increments
        if key_state.get('shoulder_pan+'):
            self.target_positions["shoulder_pan"] += self.degree_step
        if key_state.get('shoulder_pan-'):
            self.target_positions["shoulder_pan"] -= self.degree_step
        if key_state.get('wrist_roll+'):
            self.target_positions["wrist_roll"] += self.degree_step
        if key_state.get('wrist_roll-'):
            self.target_positions["wrist_roll"] -= self.degree_step
        if key_state.get('gripper+'):
            self.target_positions["gripper"] += self.degree_step
        if key_state.get('gripper-'):
            self.target_positions["gripper"] -= self.degree_step
        if key_state.get('pitch+'):
            self.pitch += self.degree_step
        if key_state.get('pitch-'):
            self.pitch -= self.degree_step

        # XY plane (IK)
        moved = False
        if key_state.get('x+'):
            self.current_x += self.xy_step
            moved = True
        if key_state.get('x-'):
            self.current_x -= self.xy_step
            moved = True
        if key_state.get('y+'):
            self.current_y += self.xy_step
            moved = True
        if key_state.get('y-'):
            self.current_y -= self.xy_step
            moved = True
        if moved:
            joint2, joint3 = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
            self.target_positions["shoulder_lift"] = joint2
            self.target_positions["elbow_flex"] = joint3

        # Wrist flex is always coupled to pitch and the other two
        self.target_positions["wrist_flex"] = (
            -self.target_positions["shoulder_lift"]
            -self.target_positions["elbow_flex"]
            + self.pitch
        )

    def p_control_action(self, robot):
        obs = robot.get_observation()
        current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        action = {}
        for j in self.target_positions:
            error = self.target_positions[j] - current[j]
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
        return action

def busy_wait(seconds):
    if platform.system() == "Darwin" or platform.system() == "Windows":
        # On Mac and Windows, `time.sleep` is not accurate and we need to use this while loop trick,
        # but it consumes CPU cycles.
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        # On Linux time.sleep is accurate
        if seconds > 0:
            time.sleep(seconds)


def xlerobot_record_loop(
    robot,
    events,
    fps,
    dataset,
    keyboard,
    left_arm,
    right_arm,
    head_control,
    control_time_s,
    single_task,
    display_data,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
):
    """
    Custom record_loop for xlerobot keyboard control.
    
    This function handles keyboard input, computes actions for both arms and head, then records data.
    """
    timestamp = 0
    start_episode_t = time.perf_counter()
    
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Process observation
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix="observation")

        # Get keyboard input
        keyboard_action = keyboard.get_action()
        pressed_keys = set(keyboard_action.keys()) if keyboard_action else set()
        left_key_state = {action: (key in pressed_keys) for action, key in LEFT_KEYMAP.items()}
        right_key_state = {action: (key in pressed_keys) for action, key in RIGHT_KEYMAP.items()}

        # Handle reset for left arm
        if left_key_state.get('reset'):
            # Reset left arm to zero position
            left_arm.target_positions = {
                'shoulder_pan': 0.0,
                'shoulder_lift': 0.0,
                'elbow_flex': 0.0,
                'wrist_flex': 0.0,
                'wrist_roll': 0.0,
                'gripper': 0.0
            }
            left_arm.current_x = 0.1629
            left_arm.current_y = 0.1131
            left_arm.pitch = 0.0

        # Handle reset for right arm
        if right_key_state.get('reset'):
            # Reset right arm to zero position
            right_arm.target_positions = {
                'shoulder_pan': 0.0,
                'shoulder_lift': 0.0,
                'elbow_flex': 0.0,
                'wrist_flex': 0.0,
                'wrist_roll': 0.0,
                'gripper': 0.0
            }
            right_arm.current_x = 0.1629
            right_arm.current_y = 0.1131
            right_arm.pitch = 0.0

        # Handle reset for head motors
        if '?' in pressed_keys:
            head_control.target_positions = head_control.zero_pos.copy()

        # Update arm and head control based on keyboard input
        left_arm.handle_keys(left_key_state)
        right_arm.handle_keys(right_key_state)
        head_control.handle_keys(left_key_state)  # Head controlled by left arm keymap

        # Get actions from arms and head
        left_action = left_arm.p_control_action(robot)
        right_action = right_arm.p_control_action(robot)
        head_action = head_control.p_control_action(robot)

        # Base action from keyboard
        # Convert set to numpy array for _from_keyboard_to_base_action
        # _from_keyboard_to_base_action expects an array of key characters
        keyboard_keys = np.array(list(pressed_keys))
        base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}

        # Combine all actions
        act = {**left_action, **right_action, **head_action, **base_action}

        # Process teleop action
        act_processed_teleop = teleop_action_processor((act, obs))

        # Process robot action
        robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        # Send action to robot
        _sent_action = robot.send_action(robot_action_to_send)

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, act_processed_teleop, prefix="action")
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=obs_processed, action=act_processed_teleop)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


def main():
    parser = argparse.ArgumentParser(description="Record datasets for xlerobot robot")
    parser.add_argument("--robot_id", type=str, default="joyandai_xlerobot", help="Robot ID")
    parser.add_argument("--remote_ip", type=str, default="192.168.200.104", help="Remote robot IP address")
    parser.add_argument("--num_episodes", type=int, default=NUM_EPISODES, help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=FPS, help="Recording frame rate")
    parser.add_argument("--episode_time_s", type=int, default=EPISODE_TIME_SEC, help="Recording time per episode (seconds)")
    parser.add_argument("--reset_time_s", type=int, default=RESET_TIME_SEC, help="Reset time (seconds)")
    parser.add_argument("--task_description", type=str, default=TASK_DESCRIPTION, help="Task description")
    parser.add_argument("--repo_id", type=str, default=None, required=True, help="HuggingFace dataset repository ID")
    parser.add_argument("--display_data", action="store_true", help="Display data visualization")
    parser.add_argument("--verbose", action="store_true", help="Show detailed logs")

    args = parser.parse_args()

    # Initialize logging
    import logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create robot and keyboard configuration
    robot_config = XLerobotClientConfig(remote_ip=args.remote_ip, id=args.robot_id)
    keyboard_config = KeyboardTeleopConfig()

    # Initialize robot and keyboard
    robot = XLerobotClient(robot_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # Create processors
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Configure dataset features
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # Connect robot and keyboard
    robot.connect()
    keyboard.connect()

    # Create dataset
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Initialize keyboard listener and visualization
    listener, events = init_keyboard_listener()
    if args.display_data:
        init_rerun(session_name="xlerobot_record")

    if not robot.is_connected or not keyboard.is_connected:
        raise ValueError("Robot or keyboard is not connected!")

    # Initialize arm and head control
    obs = robot.get_observation()
    kin_left = SO101Kinematics()
    kin_right = SO101Kinematics()
    left_arm = SimpleTeleopArm(kin_left, LEFT_JOINT_MAP, obs, prefix="left")
    right_arm = SimpleTeleopArm(kin_right, RIGHT_JOINT_MAP, obs, prefix="right")
    head_control = SimpleHeadControl(obs)

    print("Starting recording loop...")
    recorded_episodes = 0
    while recorded_episodes < args.num_episodes and not events["stop_recording"]:
        log_say(f"Recording episode {recorded_episodes}")

        # Main recording loop
        xlerobot_record_loop(
            robot=robot,
            events=events,
            fps=args.fps,
            dataset=dataset,
            keyboard=keyboard,
            left_arm=left_arm,
            right_arm=right_arm,
            head_control=head_control,
            control_time_s=args.episode_time_s,
            single_task=args.task_description,
            display_data=args.display_data,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        # Reset environment (if not stopping or re-recording)
        if not events["stop_recording"] and (
            (recorded_episodes < args.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Resetting environment")
            xlerobot_record_loop(
                robot=robot,
                events=events,
                fps=args.fps,
                dataset=None,  # Don't record data during reset
                keyboard=keyboard,
                left_arm=left_arm,
                right_arm=right_arm,
                head_control=head_control,
                control_time_s=args.reset_time_s,
                single_task=args.task_description,
                display_data=args.display_data,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

        if events["rerecord_episode"]:
            log_say("Re-recording episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        # Save episode
        dataset.save_episode()
        recorded_episodes += 1

    # Cleanup
    log_say("Stopping recording")
    robot.disconnect()
    keyboard.disconnect()
    listener.stop()
    # By default, don't push. Uncomment when needed
    # dataset.push_to_hub()


if __name__ == "__main__":
    main()