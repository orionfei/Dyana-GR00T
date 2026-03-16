import argparse
import asyncio
import json
import os
import random
from collections import deque
from pathlib import Path
from typing import Deque

import numpy as np
import pandas as pd
import torch

from unity_server import UnityServer

from gr00t.data.transform.video import VideoCrop
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


TASK_PROMPT_MAP = {
    "circular": "Grab the object in the video that is making a circular motion",
    "linear": "Grab the object in the video that is making a straight motion",
    "harmonic": "Grab the object in the video that is doing simple harmonic motion",
}

DEFAULT_MODEL_ROOT = Path("/data1/yfl_data/Dyana-GR00T/checkpoints/dyana_hand_task_lora")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_unity_meta(dataset_path: str, traj_id: int) -> dict:
    meta_data_path = os.path.join(dataset_path, "unity_meta", f"episode_{traj_id:06d}.json")
    with open(meta_data_path, "r") as f:
        return json.load(f)


def get_trajectory_data(dataset_path: str, traj_id: int) -> pd.DataFrame:
    data_path = os.path.join(dataset_path, "data", "chunk-000", f"episode_{traj_id:06d}.parquet")
    return pd.read_parquet(data_path)


def evaluate_traj2(traj: np.ndarray) -> dict:
    n = traj.shape[0]
    if n <= 1:
        return {"smoothness_var": 1.0, "linearity": 0.0}
    if n == 2:
        overall_vec = traj[1] - traj[0]
        if np.linalg.norm(overall_vec) < 1e-8:
            return {"smoothness_var": 1.0, "linearity": 0.0}
        return {"smoothness_var": 1.0, "linearity": 1.0}

    diffs = np.linalg.norm(traj[1:] - traj[:-1], axis=1)
    diffs = diffs[diffs > 1e-4]
    if diffs.size == 0:
        return {"smoothness_var": 1.0, "linearity": 0.0}

    mean_step = np.mean(diffs)
    std_step = np.std(diffs)
    smoothness_var = 1.0 if mean_step < 1e-8 else 1 / (1 + std_step / mean_step)

    overall_vec = traj[-1] - traj[0]
    overall_norm = np.linalg.norm(overall_vec)
    if overall_norm < 1e-8:
        linearity = 0.0
    else:
        overall_unit = overall_vec / overall_norm
        segs = traj[1:] - traj[:-1]
        seg_norms = np.linalg.norm(segs, axis=1, keepdims=True)
        valid = seg_norms[:, 0] > 1e-8
        if not np.any(valid):
            linearity = 0.0
        else:
            segs_unit = segs[valid] / seg_norms[valid]
            cos_sims = segs_unit @ overall_unit
            linearity = float(np.mean(cos_sims))

    return {"smoothness_var": float(smoothness_var), "linearity": float(linearity)}


def parse_task_type(obs: dict, language_key: str, fallback: str) -> str:
    if language_key in obs:
        raw = obs[language_key]
        if isinstance(raw, np.ndarray):
            if raw.ndim == 0:
                return str(raw.item())
            if raw.size > 0:
                return str(raw.reshape(-1)[0])
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            return str(raw[0])
        if isinstance(raw, str):
            return raw
    return fallback


def resolve_default_model_path() -> str:
    checkpoint_dirs = sorted(DEFAULT_MODEL_ROOT.glob("checkpoint-*"))
    if checkpoint_dirs:
        return str(checkpoint_dirs[-1])
    return str(DEFAULT_MODEL_ROOT)


def build_policy(
    model_path: str,
    embodiment_tag: str,
    data_config_name: str,
    device: str,
) -> Gr00tPolicy:
    data_config = DATA_CONFIG_MAP[data_config_name]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )
    policy.modality_transform.eval()
    return policy


def sync_policy_video_resolution(policy: Gr00tPolicy, video_key: str, image: np.ndarray) -> None:
    """Align policy video metadata with runtime eval image resolution."""
    if image.ndim != 3:
        raise ValueError(f"Expected image to be [H,W,C], got {image.shape}")
    height, width = int(image.shape[0]), int(image.shape[1])
    runtime_resolution = (width, height)

    video_sub_key = video_key.split(".", 1)[1]
    expected_resolution = tuple(policy.metadata.modalities.video[video_sub_key].resolution)
    if runtime_resolution == expected_resolution:
        return

    policy.metadata.modalities.video[video_sub_key].resolution = runtime_resolution
    # If crop transform inferred fixed H/W from old metadata, force re-inference.
    for transform in policy.modality_transform.transforms:
        if isinstance(transform, VideoCrop):
            transform.height = None
            transform.width = None
    policy.modality_transform.set_metadata(policy.metadata)
    print(
        f"Synced policy video resolution for {video_key}: "
        f"{expected_resolution} -> {runtime_resolution}"
    )


def run_eval_with_gr00t(
    server: UnityServer,
    policy: Gr00tPolicy,
    dataset_path: str,
    traj_id: int,
    repeat_num: int,
    eval_window: int,
    metrics_json_path: str,
    traj_store_path: str,
    output_action_dim: int = 18,
):
    unity_meta = get_unity_meta(dataset_path, traj_id)
    gt_actions = np.stack(get_trajectory_data(dataset_path, traj_id)["action"].to_numpy())
    steps = gt_actions.shape[0]

    modality_config = policy.modality_config
    video_key = modality_config["video"].modality_keys[0]
    state_key = modality_config["state"].modality_keys[0]
    action_key = modality_config["action"].modality_keys[0]
    language_key = modality_config["language"].modality_keys[0]
    obs_horizon = len(modality_config["state"].delta_indices)

    state_sub_key = state_key.split(".", 1)[1]
    action_sub_key = action_key.split(".", 1)[1]
    expected_state_dim = int(policy.metadata.modalities.state[state_sub_key].shape[0])
    model_action_dim = int(policy.metadata.modalities.action[action_sub_key].shape[0])
    if model_action_dim != output_action_dim:
        raise ValueError(
            f"Strict eval requires model native output dim == requested output dim. "
            f"model_action_dim={model_action_dim}, output_action_dim={output_action_dim}, action_key={action_key}"
        )

    success = server.send_start_episode_sync(
        episode_id=unity_meta["episode"],
        task_type=unity_meta["task_type"],
        repeat_num=repeat_num,
        steps=steps,
        start_frame_idx=0,
        windowSize=eval_window,
    )
    if not success:
        print(f"start_episode failed for traj_id={traj_id}, repeat={repeat_num}")
        return

    video_hist: Deque[np.ndarray] = deque(maxlen=obs_horizon)
    state_hist: Deque[np.ndarray] = deque(maxlen=obs_horizon)
    trajectory = []

    for step in range(0, steps, eval_window):
        obs = server.get_obs(block=True)
        if obs is None:
            print(f"obs timeout at traj_id={traj_id}, step={step}")
            return

        image = np.asarray(obs[video_key])
        state = np.asarray(obs[state_key], dtype=np.float64)

        if image.ndim == 4:
            image = image[0]
        if state.ndim == 2:
            state = state[0]

        # Unity obs may be resized (e.g. 256x256). Keep policy metadata in sync.
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
            language_key: np.array([prompt] * obs_horizon),
        }

        action_pred = policy.get_action(model_obs)
        print(action_pred)
        if action_key not in action_pred:
            raise KeyError(f"Missing action key '{action_key}' in policy output: {list(action_pred.keys())}")
        actions = np.asarray(action_pred[action_key])

        if actions.ndim == 3:
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError(f"Expected actions to be 2D [T, D], got shape {actions.shape}")
        if actions.shape[1] != output_action_dim:
            raise ValueError(
                f"Action dim mismatch for {action_key}: got {actions.shape[1]}, expected {output_action_dim}"
            )

        remaining = steps - step
        if actions.shape[0] < min(eval_window, remaining):
            raise ValueError(
                f"Action horizon too short: got {actions.shape[0]}, "
                f"required at least {min(eval_window, remaining)}"
            )
        chunk_actions = actions[: min(eval_window, remaining)].astype(np.float32)
        assert chunk_actions.shape[1] == output_action_dim, (
            f"Expected output action dim {output_action_dim}, got {chunk_actions.shape}"
        )
        server.send_action_data_sync(chunk_actions)
        trajectory.extend(chunk_actions)
        print(
            f"traj={traj_id} repeat={repeat_num} step={step}/{steps} "
            f"obs={model_obs[video_key].shape}/{model_obs[state_key].shape} "
            f"action={chunk_actions.shape} (model_dim={model_action_dim}, output_dim={output_action_dim})"
        )

    metrics_unity = server.get_metrics_from_unity(block=True)
    if metrics_unity is None:
        metrics_unity = {}
    success_index = int(metrics_unity.get("successIndex", len(trajectory)))
    metrics_unity.pop("successIndex", None)

    trajectory_for_save = np.asarray(trajectory, dtype=np.float32)
    trajectory_for_eval = trajectory_for_save[:success_index]
    if trajectory_for_eval.ndim == 1:
        if trajectory_for_eval.size == 0:
            trajectory_for_eval = trajectory_for_eval.reshape(0, output_action_dim)
        else:
            trajectory_for_eval = trajectory_for_eval.reshape(1, -1)
    metrics = evaluate_traj2(trajectory_for_eval)

    os.makedirs(traj_store_path, exist_ok=True)
    np.save(os.path.join(traj_store_path, f"traj_{traj_id}:repeat{repeat_num}.npy"), trajectory_for_save)

    results = {
        "traj_id": traj_id,
        "repeat_num": repeat_num,
        "episode": int(unity_meta["episode"]),
        "task_type": unity_meta["task_type"],
        **metrics,
        **metrics_unity,
        "successIndex / total_frames": f"{success_index} / {steps}",
    }
    os.makedirs(os.path.dirname(metrics_json_path), exist_ok=True)
    with open(metrics_json_path, "a") as f:
        f.write(json.dumps(results) + "\n")

    print(
        f"traj={traj_id} repeat={repeat_num} done | "
        f"smoothness={metrics['smoothness_var']:.4f} linearity={metrics['linearity']:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GR00T policy with UnityServer")
    parser.add_argument("--model-path", default=resolve_default_model_path())
    parser.add_argument("--embodiment-tag", default="dyana_hand_task")
    parser.add_argument("--data-config", default="dyana_lora_11f_18d")
    parser.add_argument(
        "--dataset-path",
        default="/data1/yfl_data/Dyana-GR00T/dyana_data",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--resize-width", type=int, default=256)
    parser.add_argument("--resize-height", type=int, default=256)
    parser.add_argument("--traj-start", type=int, default=982)
    parser.add_argument("--traj-end", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--eval-window", type=int, default=10)
    parser.add_argument("--output-action-dim", type=int, default=18)
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--connect-timeout-sec", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    policy = build_policy(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        data_config_name=args.data_config,
        device=args.device,
    )
    print(
        f"policy loaded | model={args.model_path} tag={args.embodiment_tag} "
        f"data_config={args.data_config} device={args.device}"
    )

    additional_info_data = os.path.basename(args.dataset_path).replace("_unity", "")
    additional_info_model = (
        args.model_path.replace("/", "_").replace(":", "_").replace("\\", "_")
    )
    traj_store_path = os.path.join(
        args.output_dir,
        "trajectories_gr00t",
        f"{additional_info_data}:{additional_info_model}",
    )
    metrics_json_path = os.path.join(
        args.output_dir,
        f"results_gr00t_{additional_info_data}:{additional_info_model}.jsonl",
    )
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(traj_store_path, exist_ok=True)

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

    loop.run_until_complete(server.start())
    print(f"waiting for Unity client at ws://{args.host}:{args.port}")
    if not server.wait_for_connection_sync(timeout=args.connect_timeout_sec):
        print("Unity client connection timeout")
        return

    print(
        f"start eval | traj=[{args.traj_start},{args.traj_end}) repeat={args.repeat} "
        f"eval_window={args.eval_window}"
    )
    try:
        for traj_id in range(args.traj_start, args.traj_end):
            for repeat_num in range(args.repeat):
                run_eval_with_gr00t(
                    server=server,
                    policy=policy,
                    dataset_path=args.dataset_path,
                    traj_id=traj_id,
                    repeat_num=repeat_num,
                    eval_window=args.eval_window,
                    metrics_json_path=metrics_json_path,
                    traj_store_path=traj_store_path,
                    output_action_dim=args.output_action_dim,
                )
    finally:
        loop.run_until_complete(server.stop())

    print(f"evaluation complete | metrics={metrics_json_path} trajectories={traj_store_path}")


if __name__ == "__main__":
    main()
