import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"  # 重要！！设置为确定输出模式
import sys
import json
import time
import torch
import numpy as np
import random

SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
print(f"Using seed: {SEED}")

from PIL import Image
import cv2
from typing import List, Dict
import pandas as pd
import asyncio

from openpi.training import config as _config
from openpi.policies import policy_config

from unity_server import UnityServer

# DATASET_PATH = "/data1/bch_data/Dataset/unity_test_0.75_cam_18dim"  # 在哪个数据集上测试
DATASET_PATH = "/mnt/sdc/bch/forBenchmark/Isaac-GR00T/demo_data/unity_test_1.0_cam_18dim"  # 在哪个数据集上测试

# For pi0.5
# CONFIG_NAME = "pi05_dyana_low_mem_finetune"
# CKPT_DIR = "ckpt_3w/pi05_dyana_low_mem_finetune/my_experiment/14999"

# For pi0
# CONFIG_NAME = "pi0_dyana_low_mem_finetune"
# CKPT_DIR = "ckpt_3w/pi0_dyana_low_mem_finetune/my_experiment/14999"

# For pi0 all data
CONFIG_NAME = "pi0_dyana_all_low_mem_finetune"
CKPT_DIR = "ckpt_14w/pi0_dyana_all_low_mem_finetune/my_experiment/86999"

# For pi05 all data
CONFIG_NAME = "pi05_dyana_all_low_mem_finetune"
CKPT_DIR = "ckpt_14w/pi05_dyana_all_low_mem_finetune/my_experiment/65999"


# For pi05 trained on 216
CONFIG_NAME = "pi05_dyana_all_low_mem_finetune"
CKPT_DIR = "ckpt_14w_new/pi05_dyana_all_low_mem_finetune/my_experiment/35999"

EVALUATION_OUTPUT_PATH = "evaluation_results"
TRAJ_IDS = range(982, 1000)


class DeterministicNoiseGenerator:
    """
    确定性噪声生成器：使用固定的随机种子生成噪声，确保每次推理的噪声是确定性的
    对于相同的traj_id和step，总是生成相同的噪声
    """
    def __init__(self, base_seed: int = SEED):
        self.base_seed = base_seed
        self._cache = {}  # 缓存已生成的噪声
    
    def generate_noise(self, action_horizon: int, action_dim: int, 
                       traj_id: int | None = None, step: int | None = None) -> np.ndarray:
        """
        生成确定性噪声
        
        Args:
            action_horizon: 动作序列长度
            action_dim: 动作维度
            traj_id: 轨迹ID（可选，用于生成确定性噪声）
            step: 步数（可选，用于生成确定性噪声）
        
        Returns:
            np.ndarray: 噪声数组，形状为 (action_horizon, action_dim)
        """
        # 如果提供了traj_id和step，使用它们生成确定性种子
        if traj_id is not None and step is not None:
            # 使用traj_id和step生成一个唯一的种子
            # 使用简单的组合确保相同输入产生相同种子（不依赖hash函数）
            # 使用大质数来混合值，避免冲突
            noise_seed = (self.base_seed * 1000003 + traj_id * 1009 + step * 17) % (2**31)
            cache_key = (noise_seed, action_horizon, action_dim)
            
            # 检查缓存
            if cache_key in self._cache:
                return self._cache[cache_key].copy()
            
            # 生成新的噪声
            rng = np.random.RandomState(noise_seed)
            noise = rng.randn(action_horizon, action_dim).astype(np.float32)
            self._cache[cache_key] = noise.copy()
            return noise
        else:
            # 如果没有提供traj_id和step，使用顺序生成（向后兼容）
            # 这种情况下每次调用都会生成不同的噪声
            rng = np.random.RandomState(self.base_seed)
            # 使用一个简单的计数器来确保每次调用生成不同的噪声
            # 但这不够理想，建议总是提供traj_id和step
            noise = rng.randn(action_horizon, action_dim).astype(np.float32)
            return noise
    
    def reset(self, base_seed: int = SEED):
        """重置生成器和缓存"""
        self.base_seed = base_seed
        self._cache = {}


def get_unity_meta(traj_id: int):
    meta_data_path = os.path.join(DATASET_PATH, "unity_meta", f"episode_{traj_id:06d}.json")
    with open(meta_data_path, "r") as f:
        meta_data = json.load(f)
    return meta_data


def get_trajectory_data(traj_id: int):
    data_path = os.path.join(DATASET_PATH, "data", "chunk-000", f"episode_{traj_id:06d}.parquet")
    return pd.read_parquet(data_path)


def get_episode_info(traj_id: int):
    episode_info_path = os.path.join(DATASET_PATH, "meta", "episodes.jsonl")
    with open(episode_info_path, "r") as f:
        episode_info = [json.loads(line) for line in f]
    return episode_info[traj_id]


def test_determinism(policy, image: np.ndarray, state: np.ndarray, prompt: str, 
                     action_horizon: int, action_dim: int,
                     tolerance: float = 1e-5, num_tests: int = 3):
    """
    测试模型输出的确定性：用相同的输入和相同的噪声连续推理多次，检查输出是否在误差范围内相同
    
    Args:
        policy: Pi模型policy对象
        image: 输入图像 (H, W, 3) 或 (1, H, W, 3)
        state: 输入状态 (action_dim,) 或 (1, action_dim)
        prompt: 文本提示
        action_horizon: 动作序列长度
        action_dim: 动作维度
        tolerance: 允许的最大误差（L2距离）
        num_tests: 测试次数（默认3次）
    
    Returns:
        dict: {
            "is_deterministic": bool,  # 是否确定性
            "max_diff": float,  # 最大差异
            "all_actions": List[np.ndarray],  # 所有测试的输出动作
            "pairwise_diffs": List[float]  # 每对输出之间的差异
        }
    """
    all_actions = []
    
    # 生成固定的噪声用于测试（使用固定的traj_id和step）
    noise_gen = DeterministicNoiseGenerator(base_seed=SEED)
    # test_noise = noise_gen.generate_noise(action_horizon, action_dim, traj_id=0, step=0)
    test_noise = noise_gen.generate_noise(action_horizon, 32, traj_id=0, step=0)


    for i in range(num_tests):
        example = {
            "observation/image": image,
            "observation/state": state,
            "prompt": prompt,
        }
        
        # 使用相同的噪声进行推理
        out = policy.infer(example, noise=test_noise)
        actions = out["actions"]
        all_actions.append(actions.copy())
    
    # 计算所有输出之间的差异
    pairwise_diffs = []
    for i in range(len(all_actions)):
        for j in range(i + 1, len(all_actions)):
            diff = np.linalg.norm(all_actions[i] - all_actions[j])
            pairwise_diffs.append(diff)
    
    max_diff = max(pairwise_diffs) if pairwise_diffs else 0.0
    is_deterministic = max_diff < tolerance
    
    # print(f"all_actions: {all_actions}")
    return {
        "is_deterministic": is_deterministic,
        "max_diff": float(max_diff),
        "all_actions": all_actions,
        "pairwise_diffs": [float(d) for d in pairwise_diffs]
    }


def run_inference_pi(policy, image: np.ndarray, state: np.ndarray, prompt: str, 
                     noise: np.ndarray | None = None):
    """
    适配Pi系列模型的推理函数
    
    Args:
        policy: Pi模型policy对象
        image: 输入图像 (H, W, 3) 或 (1, H, W, 3)
        state: 输入状态 (action_dim,) 或 (1, action_dim)
        prompt: 文本提示
        noise: 可选的噪声数组，形状为 (action_horizon, action_dim)。如果为None，模型会随机生成
    
    Returns:
        np.ndarray: 预测的动作序列 (action_horizon, action_dim)
    """
    example = {
        "observation/image": image,
        "observation/state": state,
        "prompt": prompt,
    }
    
    out = policy.infer(example, noise=noise)
    actions = out["actions"]
    return actions


def evaluate_traj2(traj: np.ndarray):
    """
    评估轨迹的平滑度与直线性
    
    Args:
        traj (np.ndarray): N x 3 的轨迹坐标 (N >= 3)
    
    Returns:
        dict: {
            "smoothness_var": 基于方差/标准差的平滑度分数 (越接近1越平滑),
            "linearity": 基于余弦相似度的直线性分数 (越接近1越直)
        }
    """
    N = traj.shape[0]
    
    # --------------------------- 0/1个点 ---------------------------
    if N <= 1:
        return {
            "smoothness_var": 1.0,   # 没有运动，定义为最平滑
            "linearity": 0.0         # 无法定义直线性
        }
    
    # --------------------------- 2个点 ---------------------------
    if N == 2:
        overall_vec = traj[1] - traj[0]
        overall_norm = np.linalg.norm(overall_vec)
        if overall_norm < 1e-8:
            return {
                "smoothness_var": 1.0,  # 两点重合，相当于静止
                "linearity": 0.0
            }
        else:
            return {
                "smoothness_var": 1.0,  # 只有一步，认为平滑
                "linearity": 1.0        # 两点必然成直线
            }
    
    # --------------------------- N >= 3 ---------------------------
    # ---------- （1） 速度（相邻帧位移） ----------
    diffs = np.linalg.norm(traj[1:] - traj[:-1], axis=1)  # N-1
    diffs = diffs[diffs > 1e-4]
    mean_step = np.mean(diffs)
    std_step = np.std(diffs)
    
    # 避免除0：若轨迹静止，平滑度记为1
    smoothness_var = 1.0 if mean_step < 1e-8 else 1 / (1 + std_step / mean_step)
    
    # ---------- （2） 直线性（余弦相似度） ----------
    overall_vec = traj[-1] - traj[0]
    overall_norm = np.linalg.norm(overall_vec)
    if overall_norm < 1e-8:
        linearity = 0.0  # 起终点相同，不算直线
    else:
        overall_unit = overall_vec / overall_norm
        segs = traj[1:] - traj[:-1]
        seg_norms = np.linalg.norm(segs, axis=1, keepdims=True)
        valid = seg_norms[:, 0] > 1e-8
        segs_unit = segs[valid] / seg_norms[valid]
        cos_sims = segs_unit @ overall_unit
        linearity = np.mean(cos_sims)  # 平均余弦相似度
    
    return {
        "smoothness_var": float(smoothness_var),
        "linearity": float(linearity)
    }


def run_eval_with_pi(
    server: UnityServer,
    policy,
    traj_id: int,
    repeat_num: int,
    action_horizon: int = 16,
    model_action_horizon: int = 50,  # 模型训练时的action_horizon，用于生成噪声
    action_dim: int = 18,
    noise_generator: DeterministicNoiseGenerator | None = None,
    metrics_json_path: str = "evaluation_results/results_pi.jsonl",
    traj_store_path: str = "evaluation_results/trajectories_pi"
):
    """
    使用Pi模型进行评估
    
    Args:
        server: UnityServer实例
        policy: Pi模型policy对象
        traj_id: 轨迹ID
        repeat_num: 重复次数
        action_horizon: 评估时使用的动作序列长度（发送给Unity的）
        model_action_horizon: 模型训练时的动作序列长度（用于生成噪声）
        action_dim: 动作维度
        noise_generator: 确定性噪声生成器，如果为None则不使用固定噪声
        metrics_json_path: 指标保存路径
        traj_store_path: 轨迹保存路径
    """
    unity_meta = get_unity_meta(traj_id)
    gt_actions = np.stack(get_trajectory_data(traj_id)["action"].to_numpy())
    steps = gt_actions.shape[0]
    print(f"steps: {steps}")
    
    # 建立 episode
    success = server.send_start_episode_sync(
        episode_id=unity_meta["episode"], 
        task_type=unity_meta["task_type"], 
        repeat_num=repeat_num, 
        steps=steps, 
        start_frame_idx=0,
        windowSize=action_horizon
    )
    
    if not success:
        print(f"❌ start_episode 失败：traj_id {traj_id}")
        return
    
    # 获取任务描述
    task_dict = {
        "circular": "Grab the object in the video that is making a circular motion",
        "linear": "Grab the object in the video that is making a straight motion",
        "harmonic": "Grab the object in the video that is doing simple harmonic motion"
    }
    prompt = task_dict[unity_meta["task_type"]]
    
    trajectory = []
    
    for step in range(steps):
        if step % action_horizon == 0:
            # 从Unity获取观察
            obs = server.get_obs(block=True)
            image = obs["video.ego_view"]  # (1, H, W, 3) from unity_server
            state = obs["state.left_hand"]  # (1, action_dim) from unity_server
            
            # Unity返回的格式是 (1, H, W, 3) 和 (1, action_dim)
            # Pi模型需要去掉batch维度
            if len(image.shape) == 4:
                image = image[0]  # 去掉batch维度 -> (H, W, 3)
            if len(state.shape) == 2:
                state = state[0]  # 去掉batch维度 -> (action_dim,)
            
            # 确保state是18维（如果Unity返回的是3维，需要扩展）
            if state.shape[0] == 3:
                # 如果只有3维，可能需要用零填充到18维，或者使用其他策略
                # 这里假设Unity返回的就是18维
                pass
            
            # 生成确定性噪声（如果提供了noise_generator）
            # 注意：噪声的形状应该匹配模型训练时的action_horizon，而不是评估时的
            noise = None
            if noise_generator is not None:
                # 使用模型训练时的action_horizon生成噪声
                # 传入traj_id和step确保相同traj的相同step总是使用相同的噪声
                # noise = noise_generator.generate_noise(
                #     model_action_horizon, action_dim, 
                #     traj_id=traj_id, step=step
                # )
                noise = noise_generator.generate_noise(
                    model_action_horizon, 32, 
                    traj_id=traj_id, step=step
                )
            
            # 推理（模型会输出model_action_horizon长度的动作序列）
            actions = run_inference_pi(policy, image, state, prompt, noise=noise)

            # 提取前10帧动作，因为现在train出来的是action horizon是50
            actions = actions[:10]
            print(actions[0])
            
            # 提取动作（Pi模型输出可能是(action_horizon, action_dim)，需要取前18维）
            if actions.shape[1] > 18:
                actions = actions[:, :18]
            
            # 如果剩余步数不足action_horizon，截断
            if step + action_horizon > steps:
                actions = actions[:steps - step]
            
            # 发送动作给Unity
            server.send_action_data_sync(actions)
            trajectory.extend(actions)
            print(f"✅ Step {step}/{steps} 推理成功，动作shape: {actions.shape}")
    
    # 获取Unity端评估结果
    metrics_unity = server.get_metrics_from_unity(block=True)
    successIndex = metrics_unity.get("successIndex", len(trajectory))
    metrics_unity.pop("successIndex")
    
    trajectory_for_save = np.array(trajectory)
    trajectory = trajectory_for_save[:successIndex]
    
    # 保存轨迹
    os.makedirs(traj_store_path, exist_ok=True)
    np.save(
        os.path.join(traj_store_path, f"traj_{traj_id}:repeat{repeat_num}.npy"), 
        trajectory_for_save
    )
    
    # 分析轨迹
    metrics = evaluate_traj2(trajectory)
    
    # 组装结果
    results = {
        "traj_id": traj_id,
        "repeat_num": repeat_num,
        "episode": int(unity_meta["episode"]),
        "task_type": unity_meta["task_type"],
        **metrics,
        **metrics_unity,
        "successIndex / total_frames": f"{successIndex} / {steps}",
    }
    
    os.makedirs(os.path.dirname(metrics_json_path), exist_ok=True)
    with open(metrics_json_path, "a") as f:
        f.write(json.dumps(results) + "\n")
    
    print(f"✅ Traj {traj_id} repeat {repeat_num} 完成 | Success: {metrics.get('success', False)} | Min XZ: {metrics.get('min_XZ', -1)}")


if __name__ == "__main__":
    # 初始化Pi模型
    print(f"使用配置：{CONFIG_NAME}")
    print(f"使用检查点：{CKPT_DIR}")
    
    config = _config.get_config(CONFIG_NAME)
    policy = policy_config.create_trained_policy(
        config, 
        CKPT_DIR,
        pytorch_device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    
    # 确保模型处于eval模式
    if hasattr(policy, '_model') and hasattr(policy._model, 'eval'):
        policy._model.eval()
    
    print("✅ Pi模型加载完成")
    
    # 获取模型配置
    action_horizon = config.model.action_horizon if hasattr(config.model, 'action_horizon') else 16
    action_dim = config.model.action_dim if hasattr(config.model, 'action_dim') else 18
    print(f"模型配置 - action_horizon: {action_horizon}, action_dim: {action_dim}")
    
    # 创建确定性噪声生成器
    noise_generator = DeterministicNoiseGenerator(base_seed=SEED)
    print(f"✅ 已创建确定性噪声生成器（基础种子: {SEED}）")
    
    # 测试确定性（可选）
    print("\n🔍 测试模型确定性（使用固定噪声）...")
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_state = np.random.randn(18).astype(np.float32)
    test_prompt = "Grab the object in the video that is making a circular motion"
    
    determinism_result = test_determinism(
        policy, test_image, test_state, test_prompt, 
        action_horizon=action_horizon, action_dim=action_dim
    )
    print(f"确定性测试结果：")
    print(f"  - 是否确定性: {determinism_result['is_deterministic']}")
    print(f"  - 最大差异: {determinism_result['max_diff']:.2e}")
    if not determinism_result['is_deterministic']:
        print("  ⚠️ 警告：模型输出不是完全确定性的！")
    else:
        print("  ✅ 模型输出是确定性的")
    
    # 创建UnityServer实例
    server = UnityServer(host="127.0.0.1", port=8765)
    # 创建新的事件循环（兼容 Python 3.11+）
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(server.start())
    
    # 标签信息与保存路径
    additional_info_data = DATASET_PATH.split("/")[-1].replace("_unity", "")
    additional_info_model = CKPT_DIR.split("/")[-1] if "/" in CKPT_DIR else CKPT_DIR
    if "pi0" in CONFIG_NAME and "pi05" not in CONFIG_NAME:
        model_type = "pi0"
    elif "pi05":
        model_type = "pi05"
    else:
        raise ValueError(f"Unknown model type: {CONFIG_NAME}")
    additional_info_model = f"{model_type}_{additional_info_model}"
    
    traj_store_path = os.path.join(
        EVALUATION_OUTPUT_PATH, 
        "trajectories_pi", 
        f"{additional_info_data}:{additional_info_model}"
    )
    
    metrics_json_path = os.path.join(
        EVALUATION_OUTPUT_PATH, 
        f"results_pi_{additional_info_data}:{additional_info_model}.jsonl"
    )
    
    os.makedirs(traj_store_path, exist_ok=True)
    os.makedirs(EVALUATION_OUTPUT_PATH, exist_ok=True)
    
    print(f"轨迹保存路径：{traj_store_path}")
    print(f"指标保存路径：{metrics_json_path}")
    
    print("等待 Unity client 连接...")
    if not server.wait_for_connection_sync(timeout=600):
        print("❌ Unity client 连接超时")
        exit(1)
    
    print("✅ Unity client 已连接，开始推理...")
    
    # 执行多个traj的评估
    # 注意：这里使用10作为action_horizon，因为模型训练时是50，但评估时只需要10
    eval_action_horizon = 10
    print(f"评估时使用的action horizon是：{eval_action_horizon}（模型训练时的action horizon是：{action_horizon}）")

    for traj_id in TRAJ_IDS:
        for repeat_num in range(1):
            run_eval_with_pi(
                server,
                policy,
                traj_id,
                repeat_num,
                action_horizon=eval_action_horizon,  # 评估时使用的action_horizon
                model_action_horizon=action_horizon,  # 模型训练时的action_horizon（用于生成噪声）
                action_dim=action_dim,
                noise_generator=noise_generator,  # 传入确定性噪声生成器
                metrics_json_path=metrics_json_path,
                traj_store_path=traj_store_path
            )
    
    loop.run_until_complete(server.stop())
    print("✅ 评估完成")

