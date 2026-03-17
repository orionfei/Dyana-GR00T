import argparse
import asyncio
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque

import numpy as np
import torch

from eval_gr00t import (
    TASK_PROMPT_MAP,
    build_policy,
    evaluate_traj2,
    get_trajectory_data,
    get_unity_meta,
    parse_task_type,
    set_seed,
    sync_policy_video_resolution,
)
from gr00t.rl.protocol import StepFeedback
from gr00t.rl.reward import compute_episode_rewards
from unity_server import UnityServer


TASK_TYPES = ("linear", "circular", "harmonic")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect Dyana AWBC rollouts from Unity")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--embodiment-tag", default="dyana_hand_task")
    parser.add_argument("--data-config", default="dyana_motion_crop_token_11f_18d")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="awbc_round0")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--resize-width", type=int, default=256)
    parser.add_argument("--resize-height", type=int, default=256)
    parser.add_argument("--traj-start", type=int, default=None)
    parser.add_argument("--traj-end", type=int, default=None)
    parser.add_argument("--episodes-per-task", type=int, default=60)
    parser.add_argument("--eval-window", type=int, default=10)
    parser.add_argument("--output-action-dim", type=int, default=18)
    parser.add_argument("--position-noise-std", type=float, default=0.01)
    parser.add_argument("--joint-noise-std", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--connect-timeout-sec", type=int, default=600)
    parser.add_argument("--obs-timeout-sec", type=int, default=120)
    parser.add_argument("--step-feedback-timeout-sec", type=int, default=180)
    parser.add_argument("--metrics-timeout-sec", type=int, default=180)
    return parser.parse_args()


def sample_traj_ids(
    dataset_path: Path,
    episodes_per_task: int,
    traj_start: int | None,
    traj_end: int | None,
    seed: int,
) -> list[tuple[str, int]]:
    meta_dir = dataset_path / "unity_meta"
    if not meta_dir.exists():
        raise FileNotFoundError(f"Unity meta directory not found: {meta_dir}")

    task_to_ids: dict[str, list[int]] = {task_type: [] for task_type in TASK_TYPES}
    for meta_path in sorted(meta_dir.glob("episode_*.json")):
        traj_id = int(meta_path.stem.split("_")[-1])
        if traj_start is not None and traj_id < traj_start:
            continue
        if traj_end is not None and traj_id >= traj_end:
            continue
        with open(meta_path, "r", encoding="utf-8") as handle:
            unity_meta = json.load(handle)
        task_type = str(unity_meta["task_type"])
        if task_type in task_to_ids:
            task_to_ids[task_type].append(traj_id)

    rng = np.random.default_rng(seed)
    sampled: list[tuple[str, int]] = []
    for task_type in TASK_TYPES:
        candidates = sorted(task_to_ids[task_type])
        if not candidates:
            raise ValueError(f"No trajectories found for task_type={task_type}")
        replace = len(candidates) < episodes_per_task
        selected = rng.choice(candidates, size=episodes_per_task, replace=replace)
        sampled.extend((task_type, int(traj_id)) for traj_id in selected.tolist())

    rng.shuffle(sampled)
    return sampled


def drain_server_queues(server: UnityServer) -> None:
    while server.get_obs(block=False) is not None:
        pass
    while server.get_metrics_from_unity(block=False) is not None:
        pass
    while server.get_step_feedback(block=False) is not None:
        pass


def apply_exploration_noise(
    chunk_actions: np.ndarray,
    position_noise_std: float,
    joint_noise_std: float,
    action_min: np.ndarray,
    action_max: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    noisy_actions = chunk_actions.astype(np.float32, copy=True)
    if noisy_actions.shape[1] >= 3 and position_noise_std > 0:
        noisy_actions[:, :3] += rng.normal(
            loc=0.0,
            scale=position_noise_std,
            size=noisy_actions[:, :3].shape,
        ).astype(np.float32)
    if noisy_actions.shape[1] > 3 and joint_noise_std > 0:
        noisy_actions[:, 3:] += rng.normal(
            loc=0.0,
            scale=joint_noise_std,
            size=noisy_actions[:, 3:].shape,
        ).astype(np.float32)
    noisy_actions = np.clip(noisy_actions, action_min[: noisy_actions.shape[1]], action_max[: noisy_actions.shape[1]])
    return noisy_actions.astype(np.float32, copy=False)


def collect_single_episode(
    server: UnityServer,
    policy,
    dataset_path: Path,
    traj_id: int,
    repeat_num: int,
    eval_window: int,
    output_action_dim: int,
    obs_timeout_sec: int,
    step_feedback_timeout_sec: int,
    metrics_timeout_sec: int,
    position_noise_std: float,
    joint_noise_std: float,
    rng: np.random.Generator,
) -> tuple[dict, dict[str, np.ndarray]]:
    unity_meta = get_unity_meta(str(dataset_path), traj_id)
    gt_actions = np.stack(get_trajectory_data(str(dataset_path), traj_id)["action"].to_numpy())
    steps = int(gt_actions.shape[0])

    modality_config = policy.modality_config
    video_key = modality_config["video"].modality_keys[0]
    state_key = modality_config["state"].modality_keys[0]
    action_key = modality_config["action"].modality_keys[0]
    language_key = modality_config["language"].modality_keys[0]
    obs_horizon = len(modality_config["state"].delta_indices)
    action_horizon = len(modality_config["action"].delta_indices)
    if action_horizon != eval_window:
        raise ValueError(
            f"AWBC collector expects action_horizon == eval_window. "
            f"Got action_horizon={action_horizon}, eval_window={eval_window}"
        )

    state_sub_key = state_key.split(".", 1)[1]
    action_sub_key = action_key.split(".", 1)[1]
    expected_state_dim = int(policy.metadata.modalities.state[state_sub_key].shape[0])
    model_action_dim = int(policy.metadata.modalities.action[action_sub_key].shape[0])
    if model_action_dim != output_action_dim:
        raise ValueError(
            f"Model output dim mismatch: expected {output_action_dim}, got {model_action_dim}"
        )

    action_stats = policy.metadata.statistics.action[action_sub_key]
    action_min = np.asarray(action_stats.min, dtype=np.float32)
    action_max = np.asarray(action_stats.max, dtype=np.float32)

    success = server.send_start_episode_sync(
        episode_id=unity_meta["episode"],
        task_type=unity_meta["task_type"],
        repeat_num=repeat_num,
        steps=steps,
        start_frame_idx=int(unity_meta.get("start_frame_idx", 0)),
        windowSize=eval_window,
    )
    if not success:
        raise RuntimeError(f"start_episode failed for traj_id={traj_id}, repeat={repeat_num}")

    video_hist: Deque[np.ndarray] = deque(maxlen=obs_horizon)
    state_hist: Deque[np.ndarray] = deque(maxlen=obs_horizon)
    executed_trajectory: list[np.ndarray] = []

    rollout_videos = []
    rollout_states = []
    rollout_actions = []
    rollout_languages = []
    rollout_task_types = []
    feedbacks: list[StepFeedback] = []

    for step in range(0, steps, eval_window):
        obs = server.get_obs(block=True, timeout=obs_timeout_sec)
        if obs is None:
            raise TimeoutError(f"Timed out waiting for obs at traj_id={traj_id}, step={step}")

        image = np.asarray(obs[video_key])
        state = np.asarray(obs[state_key], dtype=np.float64)
        if image.ndim == 4:
            image = image[0]
        if state.ndim == 2:
            state = state[0]

        sync_policy_video_resolution(policy, video_key, image)

        state = np.asarray(state).reshape(-1)
        if state.shape[0] != expected_state_dim:
            raise ValueError(
                f"State dim mismatch for {state_key}: got {state.shape[0]}, expected {expected_state_dim}"
            )

        task_type = parse_task_type(obs, language_key, unity_meta["task_type"])
        prompt = TASK_PROMPT_MAP.get(task_type, task_type)

        video_hist.append(image)
        state_hist.append(state)
        while len(video_hist) < obs_horizon:
            video_hist.append(video_hist[0])
            state_hist.append(state_hist[0])

        model_obs = {
            video_key: np.stack(list(video_hist), axis=0),
            state_key: np.stack(list(state_hist), axis=0),
            language_key: np.asarray([prompt] * obs_horizon),
        }

        action_pred = policy.get_action(model_obs)
        if action_key not in action_pred:
            raise KeyError(f"Missing action key '{action_key}' in policy output: {list(action_pred.keys())}")
        actions = np.asarray(action_pred[action_key])
        if actions.ndim == 3:
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError(f"Expected actions to be 2D [T, D], got shape {actions.shape}")
        if actions.shape[0] < eval_window:
            raise ValueError(
                f"Action horizon too short: got {actions.shape[0]}, required at least {eval_window}"
            )
        if actions.shape[1] != output_action_dim:
            raise ValueError(
                f"Action dim mismatch for {action_key}: got {actions.shape[1]}, expected {output_action_dim}"
            )

        stored_chunk_actions = actions[:eval_window].astype(np.float32)
        remaining = steps - step
        execute_len = min(eval_window, remaining)
        executed_chunk_actions = stored_chunk_actions[:execute_len]
        noisy_actions = apply_exploration_noise(
            chunk_actions=executed_chunk_actions,
            position_noise_std=position_noise_std,
            joint_noise_std=joint_noise_std,
            action_min=action_min,
            action_max=action_max,
            rng=rng,
        )

        if not server.send_action_data_sync(noisy_actions):
            raise RuntimeError(f"send_action_data failed for traj_id={traj_id}, decision_step={len(feedbacks)}")

        feedback_payload = server.get_step_feedback_sync(
            block=True,
            timeout=step_feedback_timeout_sec,
        )
        if feedback_payload is None:
            raise TimeoutError(
                f"Timed out waiting for step_feedback at traj_id={traj_id}, decision_step={len(feedbacks)}"
            )
        feedback = StepFeedback.from_payload(feedback_payload)
        if feedback.decision_step != len(feedbacks):
            raise ValueError(
                f"Unexpected decision_step order: got {feedback.decision_step}, expected {len(feedbacks)}"
            )

        rollout_videos.append(model_obs[video_key].astype(np.uint8, copy=True))
        rollout_states.append(model_obs[state_key].astype(np.float32, copy=True))
        rollout_actions.append(stored_chunk_actions.astype(np.float32, copy=True))
        rollout_languages.append(prompt)
        rollout_task_types.append(task_type)
        feedbacks.append(feedback)
        executed_trajectory.extend(noisy_actions.astype(np.float32, copy=False))

    metrics_unity = server.get_metrics_from_unity(block=True, timeout=metrics_timeout_sec)
    if metrics_unity is None:
        raise TimeoutError(f"Timed out waiting for metrics for traj_id={traj_id}")

    rewards, returns = compute_episode_rewards(feedbacks)
    decision_steps = np.asarray([feedback.decision_step for feedback in feedbacks], dtype=np.int32)
    done_flags = np.asarray([feedback.done for feedback in feedbacks], dtype=np.bool_)
    trajectory_for_save = np.asarray(executed_trajectory, dtype=np.float32)
    success_index = int(metrics_unity.get("successIndex", len(trajectory_for_save)))
    trajectory_for_eval = trajectory_for_save[:success_index]
    if trajectory_for_eval.ndim == 1:
        if trajectory_for_eval.size == 0:
            trajectory_for_eval = trajectory_for_eval.reshape(0, output_action_dim)
        else:
            trajectory_for_eval = trajectory_for_eval.reshape(1, -1)
    traj_metrics = evaluate_traj2(trajectory_for_eval)

    rollout_arrays = {
        video_key: np.stack(rollout_videos, axis=0).astype(np.uint8),
        state_key: np.stack(rollout_states, axis=0).astype(np.float32),
        action_key: np.stack(rollout_actions, axis=0).astype(np.float32),
        language_key: np.asarray(rollout_languages),
        "task_type": np.asarray(rollout_task_types),
        "decision_step": decision_steps,
        "reward": rewards.astype(np.float32),
        "return": returns.astype(np.float32),
        "done": done_flags,
        "success": np.asarray([feedback.success for feedback in feedbacks], dtype=np.bool_),
        "current_frame_index": np.asarray(
            [feedback.current_frame_index for feedback in feedbacks], dtype=np.int32
        ),
        "successIndex": np.asarray([feedback.success_index for feedback in feedbacks], dtype=np.int32),
        "min_distance_to_target": np.asarray(
            [feedback.min_distance_to_target for feedback in feedbacks], dtype=np.float32
        ),
        "minJointToSurfaceDistance": np.asarray(
            [feedback.min_joint_to_surface_distance for feedback in feedbacks], dtype=np.float32
        ),
    }
    episode_return = float(returns[0]) if len(returns) > 0 else 0.0
    episode_metrics = {
        "traj_id": traj_id,
        "repeat_num": repeat_num,
        "episode": int(unity_meta["episode"]),
        "task_type": unity_meta["task_type"],
        "num_chunks": int(len(feedbacks)),
        "num_executed_actions": int(trajectory_for_save.shape[0]),
        "episode_return": episode_return,
        "success": bool(metrics_unity.get("success", False)),
        "waitTime": float(metrics_unity.get("waitTime", 0.0)),
        "score": float(metrics_unity.get("score", 0.0)),
        "min_XZ": float(metrics_unity.get("min_XZ", np.nan)),
        "successIndex": success_index,
        "minJointToSurfaceDistance": float(metrics_unity.get("minJointToSurfaceDistance", np.nan)),
        "smoothness_var": float(traj_metrics["smoothness_var"]),
        "linearity": float(traj_metrics["linearity"]),
    }
    return episode_metrics, rollout_arrays


def write_episode(
    run_dir: Path,
    episode_index: int,
    episode_metrics: dict,
    rollout_arrays: dict[str, np.ndarray],
) -> None:
    episode_dir = run_dir / f"episode_{episode_index:06d}"
    episode_dir.mkdir(parents=True, exist_ok=False)

    with open(episode_dir / "episode_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(episode_metrics, handle, indent=2)

    np.savez_compressed(episode_dir / "chunks.npz", **rollout_arrays)


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_path = Path(args.dataset_path)
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    policy = build_policy(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        data_config_name=args.data_config,
        device=args.device,
    )
    policy.modality_transform.eval()

    sampled_trajs = sample_traj_ids(
        dataset_path=dataset_path,
        episodes_per_task=args.episodes_per_task,
        traj_start=args.traj_start,
        traj_end=args.traj_end,
        seed=args.seed,
    )
    repeat_counters: dict[int, int] = defaultdict(int)
    rng = np.random.default_rng(args.seed + 1)

    server = UnityServer(
        host=args.host,
        port=args.port,
        resize_size=(args.resize_width, args.resize_height),
    )
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(server.start())
        print(f"waiting for Unity client at ws://{args.host}:{args.port}")
        if not server.wait_for_connection_sync(timeout=args.connect_timeout_sec):
            raise TimeoutError("Unity client connection timeout")

        metrics_jsonl = run_dir / "metrics.jsonl"
        with open(metrics_jsonl, "w", encoding="utf-8") as _:
            pass

        for episode_index, (_, traj_id) in enumerate(sampled_trajs):
            repeat_num = repeat_counters[traj_id]
            repeat_counters[traj_id] += 1
            drain_server_queues(server)
            episode_metrics, rollout_arrays = collect_single_episode(
                server=server,
                policy=policy,
                dataset_path=dataset_path,
                traj_id=traj_id,
                repeat_num=repeat_num,
                eval_window=args.eval_window,
                output_action_dim=args.output_action_dim,
                obs_timeout_sec=args.obs_timeout_sec,
                step_feedback_timeout_sec=args.step_feedback_timeout_sec,
                metrics_timeout_sec=args.metrics_timeout_sec,
                position_noise_std=args.position_noise_std,
                joint_noise_std=args.joint_noise_std,
                rng=rng,
            )
            write_episode(run_dir, episode_index, episode_metrics, rollout_arrays)
            with open(metrics_jsonl, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(episode_metrics) + "\n")
            print(
                f"collected episode {episode_index:06d} | traj={traj_id} "
                f"task={episode_metrics['task_type']} success={episode_metrics['success']} "
                f"return={episode_metrics['episode_return']:.4f}"
            )
    finally:
        loop.run_until_complete(server.stop())


if __name__ == "__main__":
    main()
