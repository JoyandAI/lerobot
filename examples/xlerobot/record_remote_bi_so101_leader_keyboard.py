"""
Record datasets for XLerobot using:
- one local `BiSO101Leader` as the dual-arm master
- one local keyboard for the mobile base
- one remote `XLerobotClient` as the dual-arm follower + base

Run robot host on the robot side first:

```bash
PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot
```

Then run this recording script on the operator side.
"""

from __future__ import annotations

import argparse
import logging
import platform
import time

import numpy as np

from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient, XLerobotClientConfig
from lerobot.teleoperators.bi_so101_leader import BiSO101Leader, BiSO101LeaderConfig
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.control_utils import init_keyboard_listener, sanity_check_dataset_robot_compatibility
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)

NUM_EPISODES = 50
FPS = 30
EPISODE_TIME_SEC = 300
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "My task description"


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


def normalize_bi_leader_action(raw_action: dict[str, float]) -> dict[str, float]:
    # `BiSO101Leader` now already returns `left_arm_*` / `right_arm_*` keys.
    # Keep support for older `left_*` / `right_*` names so this script works with both.
    return {ARM_ACTION_MAP.get(key, key): value for key, value in raw_action.items()}


def build_full_robot_action(
    raw_leader_action: dict[str, float],
    base_action: dict[str, float],
    observation: dict[str, float],
    action_features: dict[str, type],
) -> dict[str, float]:
    leader_action = normalize_bi_leader_action(raw_leader_action)

    required_arm_keys = [key for key in action_features if key.startswith(("left_arm_", "right_arm_"))]
    missing_arm_keys = [key for key in required_arm_keys if key not in leader_action]
    if missing_arm_keys:
        raise KeyError(
            "Leader action is missing required arm joints. "
            f"Missing: {missing_arm_keys}. Available keys: {sorted(raw_leader_action)}"
        )

    action = {**leader_action, **base_action}

    for key in action_features:
        if key in action:
            continue
        if key.endswith(".vel"):
            action[key] = 0.0
        elif key.endswith(".pos"):
            action[key] = float(observation[key])

    return action


def busy_wait(seconds: float) -> None:
    # Keep a more stable control rate on Windows/macOS, where time.sleep can be coarse.
    if platform.system() in {"Darwin", "Windows"}:
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    elif seconds > 0:
        time.sleep(seconds)


def make_round_prompt(
    session_episode_idx: int,
    session_total: int,
    dataset_episode_idx: int,
    *,
    rerecord: bool = False,
) -> str:
    action_text = "准备重录" if rerecord else "准备开始"
    return (
        f"{action_text}第 {session_episode_idx + 1}/{session_total} 轮，"
        f"数据集总第 {dataset_episode_idx + 1} 轮。"
    )


def clear_phase_exit_event(events: dict[str, bool]) -> None:
    # `exit_early` is a one-shot control event. Clear stale presses before entering a new phase
    # so a right-arrow used to end reset doesn't immediately terminate the next recording phase.
    events["exit_early"] = False


def record_loop(
    robot,
    leader,
    keyboard,
    events,
    fps,
    control_time_s,
    dataset,
    single_task,
    display_data,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
):
    # One loop iteration corresponds to one recorded control step.
    timestamp = 0.0
    start_episode_t = time.perf_counter()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        # The global keyboard listener can request "finish episode now".
        if events["exit_early"]:
            events["exit_early"] = False
            break

        try:
            # Read the follower robot state first so both logging and action processing
            # use the same observation snapshot for this timestep.
            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)

            observation_frame = None
            if dataset is not None:
                # Convert the processed observation dict into the dataset schema.
                observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix="observation")

            # Arms come from the physical bi-leader, while the base comes from keyboard keys.
            leader_action = leader.get_action()
            pressed_keys = np.array(list(keyboard.get_action().keys()))
            base_action = robot._from_keyboard_to_base_action(pressed_keys) or {}
            act = build_full_robot_action(
                raw_leader_action=leader_action,
                base_action=base_action,
                observation=obs,
                action_features=robot.action_features,
            )

            # Keep the standard LeRobot processing hooks even though the default processors
            act_processed_teleop = teleop_action_processor((act, obs))

            # to robot
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
            sent_action = robot.send_action(robot_action_to_send)

            if dataset is not None and observation_frame is not None:
                # Store the teleop action that corresponds to the current observation frame.
                action_frame = build_dataset_frame(dataset.features, act_processed_teleop, prefix="action")
                frame = {**observation_frame, **action_frame, "task": single_task}
                dataset.add_frame(frame)

            if display_data:
                log_rerun_data(observation=obs_processed, action=sent_action)

            pressed_key_set = set(pressed_keys.tolist())
            # Reuse the robot keyboard quit key as an additional stop signal for recording.
            if robot.teleop_keys["quit"] in pressed_key_set:
                events["discard_current_episode"] = True
                events["stop_recording"] = True
                break
        except Exception as exc:
            logger.exception("Recording interrupted, discarding current episode: %s", exc)
            print(f"录制中断，舍弃本轮数据：{exc}")
            events["discard_current_episode"] = True
            events["stop_recording"] = True
            break

        # Run the loop close to the requested recording FPS.
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)
        timestamp = time.perf_counter() - start_episode_t


def main():
    parser = argparse.ArgumentParser(description="Record datasets for XLerobot using bi-so101 leader + keyboard")
    parser.add_argument("--robot_id", type=str, default="joyandai_xlerobot", help="Robot ID")
    parser.add_argument("--remote_ip", type=str, default="192.168.200.9", help="Remote robot IP address")
    parser.add_argument("--leader_id", type=str, default="my_bi_so101_leader", help="Bi leader ID")
    parser.add_argument("--left_leader_port", type=str, default="COM8", help="Left leader serial port")
    parser.add_argument("--right_leader_port", type=str, default="COM9", help="Right leader serial port")
    parser.add_argument("--num_episodes", type=int, default=NUM_EPISODES, help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=FPS, help="Recording frame rate")
    parser.add_argument("--episode_time_s", type=int, default=EPISODE_TIME_SEC, help="Recording time per episode")
    parser.add_argument("--reset_time_s", type=int, default=RESET_TIME_SEC, help="Reset time between episodes")
    parser.add_argument("--task_description", type=str, default=TASK_DESCRIPTION, help="Task description")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace dataset repository ID")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume recording into an existing dataset. --num_episodes means additional episodes for this run.",
    )
    parser.add_argument("--display_data", action="store_true", help="Display data visualization")
    parser.add_argument("--verbose", action="store_true", help="Show detailed logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create the three devices used by this script:
    # 1) remote follower robot, 2) local bi-leader for both arms, 3) local keyboard for the base.
    robot = XLerobotClient(XLerobotClientConfig(remote_ip=args.remote_ip, id=args.robot_id))
    leader = BiSO101Leader(
        config=BiSO101LeaderConfig(
            id=args.leader_id,
            left_arm_port=args.left_leader_port,
            right_arm_port=args.right_leader_port,
        )
    )
    keyboard = KeyboardTeleop(KeyboardTeleopConfig(id="my_laptop_keyboard"))

    # These are the standard processing pipelines used throughout LeRobot record/eval flows.
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Build the dataset schema from the robot hardware specs so frames can be packed
    # into the standard LeRobot observation/action format.
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    robot.connect()
    leader.connect()
    keyboard.connect()

    if args.resume:
        dataset = LeRobotDataset(args.repo_id, batch_encoding_size=1)
        dataset.start_image_writer(num_threads=4)
        sanity_check_dataset_robot_compatibility(dataset, robot, args.fps, dataset_features)
    else:
        # Create a fresh dataset writer. Each episode is accumulated frame by frame and
        # committed with dataset.save_episode() below.
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )

    # This listener is separate from KeyboardTeleop. It is only used to control
    # episode flow: finish early, re-record, or stop the whole recording session.
    listener, events = init_keyboard_listener()
    events["discard_current_episode"] = False
    if args.display_data:
        init_rerun(session_name="xlerobot_remote_bi_so101_record")

    if not robot.is_connected or not leader.is_connected or not keyboard.is_connected:
        raise ValueError("Robot, bi leader, or keyboard is not connected.")

    print("Starting recording loop...")
    print("Base keys: i/k/j/l move, u/o rotate, n/m speed +/-, b quit")
    print("Listener keys: -> end episode, <- re-record episode, Esc stop recording")
    if args.resume:
        print(f"Resume mode: existing dataset has {dataset.num_episodes} episodes. This run will add {args.num_episodes} episodes.")
    else:
        print(f"New dataset mode: this run will record {args.num_episodes} episodes.")

    recorded_episodes = 0
    try:
        with VideoEncodingManager(dataset):
            while recorded_episodes < args.num_episodes and not events["stop_recording"]:
                clear_phase_exit_event(events)
                round_prompt = make_round_prompt(
                    session_episode_idx=recorded_episodes,
                    session_total=args.num_episodes,
                    dataset_episode_idx=dataset.num_episodes,
                )
                print(round_prompt)
                log_say(round_prompt)

                # Main recording phase: send teleop actions to the robot and write frames to the dataset.
                record_loop(
                    robot=robot,
                    leader=leader,
                    keyboard=keyboard,
                    events=events,
                    fps=args.fps,
                    control_time_s=args.episode_time_s,
                    dataset=dataset,
                    single_task=args.task_description,
                    display_data=args.display_data,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )

                if not events["stop_recording"] and (
                    (recorded_episodes < args.num_episodes - 1) or events["rerecord_episode"]
                ):
                    clear_phase_exit_event(events)
                    print("进入重置阶段，请复位环境。")
                    log_say("Resetting environment")
                    # Reset phase: operator can move the scene/robot back to a start state
                    # without writing reset motions into the dataset.
                    record_loop(
                        robot=robot,
                        leader=leader,
                        keyboard=keyboard,
                        events=events,
                        fps=args.fps,
                        control_time_s=args.reset_time_s,
                        dataset=None,
                        single_task=args.task_description,
                        display_data=args.display_data,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                    )
                    clear_phase_exit_event(events)

                if events["rerecord_episode"]:
                    rerecord_prompt = make_round_prompt(
                        session_episode_idx=recorded_episodes,
                        session_total=args.num_episodes,
                        dataset_episode_idx=dataset.num_episodes,
                        rerecord=True,
                    )
                    print(rerecord_prompt)
                    log_say(rerecord_prompt)
                    # Drop all frames collected for the current episode and try again.
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                if events["discard_current_episode"]:
                    print("已舍弃当前轮数据，之前已保存轮次保持不变。")
                    log_say("Discard current episode")
                    events["discard_current_episode"] = False
                    dataset.clear_episode_buffer()
                    break

                if events["stop_recording"]:
                    print("停止录制，舍弃当前未完成轮次，之前已保存数据保留。")
                    log_say("Discard unfinished episode")
                    dataset.clear_episode_buffer()
                    break

                # Persist the buffered frames as one complete episode.
                dataset.save_episode()
                print(
                    f"已保存第 {recorded_episodes + 1}/{args.num_episodes} 轮，"
                    f"数据集当前共有 {dataset.num_episodes} 轮。"
                )
                recorded_episodes += 1
    finally:
        log_say("Stopping recording")
        # Disconnect in reverse order of usage and stop the optional keyboard listener.
        if listener is not None:
            listener.stop()
        if keyboard.is_connected:
            keyboard.disconnect()
        if leader.is_connected:
            leader.disconnect()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
