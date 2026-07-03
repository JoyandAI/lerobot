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

from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_processors
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig, LeKiwi, LeKiwiConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
import argparse

NUM_EPISODES = 300
FPS = 30
EPISODE_TIME_SEC = 3000
RESET_TIME_SEC = 20
TASK_DESCRIPTION = "catch the block"
HF_REPO_ID = "lekiwi_test/catch_block"


def main():
    parser = argparse.ArgumentParser(description="Record datasets for lekiwi robot")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    resume = args.resume 

    # Create the robot and teleoperator configurations
    robot_config = LeKiwiClientConfig(remote_ip="192.168.200.52", id="joyandai") # remote
    # robot_config = LeKiwiConfig(port="COM3",id="my_lekiwi")     # local

    # port in Linux: /dev/ttyACM0, /dev/ttyACM1, etc.
    # port in MacOS: /dev/tty.usbmodemXXXXXXXXXXXX
    # port in Windows: COMX / COMXX
    leader_arm_config = SO101LeaderConfig(port="COM10", id="my_so101")
    keyboard_config = KeyboardTeleopConfig()

    # Initialize the robot and teleoperator
    robot = LeKiwiClient(robot_config)  # remote
    #robot = LeKiwi(robot_config) # local

    leader_arm = SO101Leader(leader_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}
    if resume:
        dataset = LeRobotDataset(
            repo_id=HF_REPO_ID,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=0,
                num_threads=4 * len(robot.cameras),
            )

        recorded_episodes = dataset.num_episodes
        print(f"[RESUME] Existing episodes: {recorded_episodes}")

    else:
        dataset = LeRobotDataset.create(
            repo_id=HF_REPO_ID,
            fps=FPS,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
            batch_encoding_size=1,
        )
        recorded_episodes = 0

    # Connect the robot and teleoperator
    # To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=LK1225XXXX`
    # where LK1225XXXX is the robot serial number
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    init_rerun(session_name="lekiwi_record")

    try:
        if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        teleop_action_processor, robot_action_processor, robot_observation_processor = (
            make_default_processors()
        )

        print("Starting record loop...")
        recorded_episodes = 0
        while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Recording episode {recorded_episodes}")

            # Main record loop
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                dataset=dataset,
                teleop=[leader_arm, keyboard],
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )

            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and (
                (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=[leader_arm, keyboard],
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=True,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # Save episode
            dataset.save_episode()
            recorded_episodes += 1
    finally:
        # Clean up
        log_say("Stop recording")
        robot.disconnect()
        leader_arm.disconnect()
        keyboard.disconnect()
        listener.stop()

        dataset.finalize()
        dataset.push_to_hub()


if __name__ == "__main__":
    main()
