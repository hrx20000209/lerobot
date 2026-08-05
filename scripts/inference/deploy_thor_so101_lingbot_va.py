#!/usr/bin/env python
"""Deploy the state-conditioned LingBot-VA LoRA on an SO-101 from Jetson Thor.

The synchronous driver deliberately mirrors ``verify_closed_loop.run_select_action``:
one fresh RGB observation and one raw measured six-joint state are passed to
``policy.select_action`` per control tick.  ``state_feedback.attach_state_feedback`` is
installed after PEFT loading and is never reimplemented here.

The asynchronous driver uses the same direct-chunk contract as LeRobot's PolicyServer:
``predict_action_chunk`` + ``set_observation_history`` +
``observation_history_size``.  Immediately before every refill it additionally calls
the state-feedback patch's ``_states_to_executed`` helper and replaces
``policy._executed_actions`` with normalized measured states.  A single inference worker
owns all mutable policy/KV-cache state.

Safety defaults are conservative:

* no motor command is sent unless ``--execute`` is present;
* live execution requires a passing offline verification JSON;
* arm targets are rate-limited against measured state, while the gripper is rate-limited
  against the previous command so a stalled jaw can still sustain the model's full close;
* absolute deployment envelopes, SO-101 gripper current/torque caps, Ctrl+C, and an
  Enter-key emergency stop are enabled.

Run ``--mode preflight`` first, then ``--mode verify``.  Only after verification passes,
run a dry robot rollout and finally add ``--execute``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOG = logging.getLogger("deploy_thor_so101_lingbot_va")
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "lingbot_va_so101"

ACTION_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
OBS_CAM_KEYS = (
    "observation.images.front",
    "observation.images.right",
    "observation.images.wrist",
)
ROBOT_CAM_KEYS = ("front", "right", "wrist")
USED_ACTION_CHANNEL_IDS = (14, 15, 16, 17, 18, 28)
DEFAULT_TASK = "go to red cube. take the red cube. go to box. put the red cube in box."

Q01 = torch.tensor([-40.3146, -104.9285, -44.1590, 59.5478, 0.5715, 8.9597])
Q99 = torch.tensor([6.3816, 44.5878, 91.8237, 96.6554, 72.5608, 35.7176])

# Conservative deployment envelopes, not claims about universal SO-101 mechanical
# limits.  Arm defaults use the demonstrated q01/q99 range.  Gripper close extends to
# zero so the state-conditioned checkpoint's verified 4--5 command is never clipped.
DEFAULT_JOINT_MIN = np.array([*Q01[:5].tolist(), 0.0], dtype=np.float32)
DEFAULT_JOINT_MAX = np.array([*Q99[:5].tolist(), 40.0], dtype=np.float32)


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def resolve_lora_checkpoint(path: str | Path) -> Path:
    """Accept either the adapter directory or its parent containing step_15000."""
    candidate = _expand(path)
    if (candidate / "adapter_model.safetensors").is_file():
        return candidate
    step = candidate / "step_15000"
    if (step / "adapter_model.safetensors").is_file():
        return step
    raise FileNotFoundError(
        f"LoRA checkpoint not found below {candidate}; expected adapter_model.safetensors "
        "either there or in step_15000/."
    )


def resolve_base_policy(path: str) -> str:
    """Resolve a local LeRobot checkpoint while preserving Hugging Face repo IDs."""
    expanded = Path(path).expanduser()
    if expanded.exists():
        root = expanded.resolve()
        missing = [name for name in ("config.json", "model.safetensors") if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{root} is not a LeRobot base-policy directory (missing {missing}). "
                "The diffusers/Wan directory belongs in --wan-pretrained-path."
            )
        return str(root)
    if "/" not in path:
        raise FileNotFoundError(f"Local base-policy path does not exist: {expanded}")
    return path


def validate_wan_assets(path: str | Path) -> Path:
    root = _expand(path)
    missing = [name for name in ("vae", "text_encoder", "tokenizer") if not (root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Wan frozen-model directory {root} is missing subdirectories: {missing}")
    return root


def parse_six_floats(value: str) -> np.ndarray:
    fields = [float(item.strip()) for item in value.split(",")]
    if len(fields) != len(ACTION_NAMES):
        raise argparse.ArgumentTypeError(f"expected six comma-separated values, got {len(fields)}")
    result = np.asarray(fields, dtype=np.float32)
    if not np.isfinite(result).all():
        raise argparse.ArgumentTypeError("all joint-limit values must be finite")
    return result


def denormalize_action(action: torch.Tensor) -> torch.Tensor:
    q01 = Q01.to(action.device, action.dtype)
    q99 = Q99.to(action.device, action.dtype)
    return (action + 1) * 0.5 * (q99 - q01) + q01


@dataclass
class StaticDatasetMetadata:
    """The feature/stats subset make_policy needs; no training dataset is required."""

    camera_height: int = 480
    camera_width: int = 640

    def __post_init__(self) -> None:
        motor_names = [f"{name}.pos" for name in ACTION_NAMES]
        self.features: dict[str, dict[str, Any]] = {
            key: {
                "dtype": "video",
                "shape": (self.camera_height, self.camera_width, 3),
                "names": ["height", "width", "channels"],
            }
            for key in OBS_CAM_KEYS
        }
        self.features["observation.state"] = {
            "dtype": "float32",
            "shape": (6,),
            "names": motor_names,
        }
        self.features["action"] = {
            "dtype": "float32",
            "shape": (6,),
            "names": motor_names,
        }
        self.stats = {
            "action": {
                "q01": Q01.clone(),
                "q99": Q99.clone(),
            }
        }


def build_policy_config(args: argparse.Namespace, base_policy: str, wan_root: Path):
    if str(EXAMPLE_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLE_DIR))
    from common import build_config

    return build_config(
        pretrained=True,
        pretrained_path=base_policy,
        attn_mode="torch",
        wan_pretrained_path=str(wan_root),
        text_encoder_device=args.text_encoder_device,
        device=args.device,
        height=128,
        width=128,
        num_inference_steps=args.video_steps,
        action_num_inference_steps=args.action_steps,
        guidance_scale=args.guidance_scale,
        action_guidance_scale=args.action_guidance_scale,
    )


def load_policy(args: argparse.Namespace, lora_checkpoint: Path, base_policy: str, wan_root: Path):
    """Mirror the verified loader: base -> text cache -> PEFT -> state feedback."""
    from lerobot.policies.factory import make_policy

    if str(EXAMPLE_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLE_DIR))
    from common import attach_text_embed_cache
    from state_feedback import attach_state_feedback

    config = build_policy_config(args, base_policy, wan_root)
    metadata = StaticDatasetMetadata(args.camera_height, args.camera_width)

    LOG.info("加载 LeRobot base policy: %s", base_policy)
    policy = make_policy(config, ds_meta=metadata)
    attach_text_embed_cache(policy)

    LOG.info("加载 LoRA/action heads: %s", lora_checkpoint)
    from peft import PeftModel

    policy = PeftModel.from_pretrained(
        policy,
        str(lora_checkpoint),
        is_trainable=False,
    ).get_base_model()
    policy.eval()

    attach_state_feedback(policy, Q01, Q99, verbose=True)
    if not hasattr(policy, "_states_to_executed"):
        raise RuntimeError("state-feedback patch was not installed")
    LOG.info(
        "状态反馈已安装；channels=%s，attn_mode=%s，去噪=%d/%d",
        list(policy.config.used_action_channel_ids),
        policy.config.attn_mode,
        policy.config.num_inference_steps,
        policy.config.action_num_inference_steps,
    )
    return policy, config


def warm_text_cache_before_robot_connect(policy, task: str) -> None:
    """Pay the CPU UMT5 cost before the robot is connected and under torque."""
    LOG.info("机器人连接前预热冻结模块和任务文本缓存（CPU UMT5 首次约需数分钟）")
    policy._ensure_frozen_modules()
    policy._maybe_init_prompt({"task": [task]})
    # reset clears per-episode prompt/KV state, but attach_text_embed_cache's memoized
    # positive and CFG-negative embeddings remain available for the live episode.
    policy.reset()
    LOG.info("文本缓存预热完成")


def raw_image_to_tensor(image: np.ndarray) -> torch.Tensor:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"camera image must be HWC RGB with 3 channels, got {array.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    if tensor.dtype == torch.uint8:
        tensor = tensor.float().div_(255.0)
    else:
        tensor = tensor.float()
        if tensor.max().item() > 1.0:
            tensor = tensor.div(255.0)
    return tensor.unsqueeze(0)


def state_from_raw_observation(raw_obs: dict[str, Any]) -> torch.Tensor:
    missing = [f"{name}.pos" for name in ACTION_NAMES if f"{name}.pos" not in raw_obs]
    if missing:
        raise KeyError(f"robot observation is missing joint keys: {missing}")
    values = [float(raw_obs[f"{name}.pos"]) for name in ACTION_NAMES]
    state = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
    if not torch.isfinite(state).all():
        raise RuntimeError(f"measured joint state contains NaN/Inf: {values}")
    return state


def build_live_batch(raw_obs: dict[str, Any], task: str) -> dict[str, Any]:
    missing = [key for key in ROBOT_CAM_KEYS if key not in raw_obs]
    if missing:
        raise KeyError(f"robot observation is missing cameras {missing}; available={sorted(raw_obs)}")
    batch = {
        policy_key: raw_image_to_tensor(raw_obs[robot_key])
        for policy_key, robot_key in zip(OBS_CAM_KEYS, ROBOT_CAM_KEYS, strict=True)
    }
    # CRITICAL: raw measured state in physical robot units, never normalized here.
    batch["observation.state"] = state_from_raw_observation(raw_obs)
    batch["task"] = [task]
    return batch


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()
    }


class SO101Controller:
    def __init__(self, args: argparse.Namespace) -> None:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

        cameras = {
            "front": OpenCVCameraConfig(
                index_or_path=args.front_camera,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
                color_mode="rgb",
                fourcc=args.camera_fourcc,
            ),
            "right": OpenCVCameraConfig(
                index_or_path=args.right_camera,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
                color_mode="rgb",
                fourcc=args.camera_fourcc,
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path=args.wrist_camera,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
                color_mode="rgb",
                fourcc=args.wrist_fourcc,
            ),
        }
        config = SOFollowerRobotConfig(
            id=args.robot_id,
            port=args.robot_port,
            cameras=cameras,
            use_degrees=True,
            # We implement an arm/state + gripper/command-aware limiter below.  The
            # generic measured-state limiter would prevent a stalled gripper from ever
            # receiving and sustaining the model's full-close target.
            max_relative_target=None,
        )
        self.robot = SOFollower(config)

    def connect(self, args: argparse.Namespace) -> None:
        self.robot.connect()
        # These are the same hardware registers SOFollower.configure sets.  Reapply the
        # requested caps explicitly so a live run logs and owns its force envelope.
        self.robot.bus.write("Max_Torque_Limit", "gripper", args.gripper_max_torque, num_retry=3)
        self.robot.bus.write(
            "Protection_Current",
            "gripper",
            args.gripper_protection_current,
            num_retry=3,
        )
        self.robot.bus.write("Overload_Torque", "gripper", args.gripper_overload_torque, num_retry=3)
        LOG.warning(
            "夹爪力限制已写入：Max_Torque=%d/1000, Protection_Current=%d/500, Overload=%d%%",
            args.gripper_max_torque,
            args.gripper_protection_current,
            args.gripper_overload_torque,
        )

    def get_observation(self) -> dict[str, Any]:
        return self.robot.get_observation()

    def send_target(self, target: np.ndarray) -> dict[str, float]:
        action = {
            f"{name}.pos": float(value) for name, value in zip(ACTION_NAMES, target.tolist(), strict=True)
        }
        return self.robot.send_action(action)

    def disconnect(self) -> None:
        if self.robot.is_connected:
            self.robot.disconnect()


@dataclass
class SafetyLimiter:
    joint_min: np.ndarray
    joint_max: np.ndarray
    max_arm_step: float
    max_gripper_command_step: float
    previous_command: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.joint_min.shape != (6,) or self.joint_max.shape != (6,):
            raise ValueError("joint limits must each contain six values")
        if np.any(self.joint_min >= self.joint_max):
            raise ValueError("every joint minimum must be below its maximum")
        if self.max_arm_step <= 0 or self.max_gripper_command_step <= 0:
            raise ValueError("rate limits must be positive")

    def apply(self, requested: np.ndarray, measured: np.ndarray) -> np.ndarray:
        requested = np.asarray(requested, dtype=np.float32).reshape(6)
        measured = np.asarray(measured, dtype=np.float32).reshape(6)
        if not np.isfinite(requested).all() or not np.isfinite(measured).all():
            raise RuntimeError(f"non-finite target/state: target={requested}, state={measured}")

        bounded = np.clip(requested, self.joint_min, self.joint_max)
        safe = bounded.copy()
        # Arm safety is tied to the actual state so a lagging/stalled arm cannot build
        # an arbitrarily large position error.
        arm_delta = np.clip(bounded[:5] - measured[:5], -self.max_arm_step, self.max_arm_step)
        safe[:5] = measured[:5] + arm_delta

        # Gripper safety is tied to the prior command, not measured state.  During a
        # successful grasp the jaw stalls near 15 while the required command remains
        # around 4; measured-state clipping would silently destroy grip force.
        gripper_reference = (
            float(measured[5]) if self.previous_command is None else float(self.previous_command[5])
        )
        gripper_delta = float(
            np.clip(
                bounded[5] - gripper_reference,
                -self.max_gripper_command_step,
                self.max_gripper_command_step,
            )
        )
        safe[5] = gripper_reference + gripper_delta
        # Do not jump an arm that starts outside the deployment envelope straight to
        # its boundary: rate limiting has priority while it returns toward the envelope.
        # Once inside, ``bounded`` guarantees it stays inside.
        safe[5] = np.clip(safe[5], self.joint_min[5], self.joint_max[5])
        self.previous_command = safe.copy()
        return safe


class SafeRobotProxy:
    """Apply the deployment limiter around the robot used by official RobotClient."""

    def __init__(
        self,
        robot,
        limiter: SafetyLimiter,
        execute: bool,
        estop: EmergencyStop,
        log_every: int,
    ) -> None:
        self._robot = robot
        self._limiter = limiter
        self._execute = execute
        self._estop = estop
        self._log_every = log_every
        self._latest_observation: dict[str, Any] | None = None
        self._step = 0

    def __getattr__(self, name: str):
        return getattr(self._robot, name)

    def get_observation(self) -> dict[str, Any]:
        if self._estop.is_set():
            raise KeyboardInterrupt("emergency stop")
        observation = self._robot.get_observation()
        self._latest_observation = observation
        return observation

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if self._estop.is_set():
            raise KeyboardInterrupt("emergency stop")
        if self._latest_observation is None:
            raise RuntimeError("收到动作前尚未取得机器人实测状态，拒绝发送")

        requested = np.asarray(
            [float(action[f"{name}.pos"]) for name in ACTION_NAMES],
            dtype=np.float32,
        )
        measured = state_from_raw_observation(self._latest_observation).squeeze(0).numpy()
        safe = self._limiter.apply(requested, measured)
        safe_action = {
            f"{name}.pos": float(value) for name, value in zip(ACTION_NAMES, safe.tolist(), strict=True)
        }
        performed = self._robot.send_action(safe_action) if self._execute else safe_action

        if self._step % self._log_every == 0:
            LOG.info(
                "official-client step=%d requested=%s measured=%s safe=%s execute=%s",
                self._step,
                np.round(requested, 2).tolist(),
                np.round(measured, 2).tolist(),
                np.round(safe, 2).tolist(),
                self._execute,
            )
        self._step += 1
        return performed

    def disconnect(self) -> None:
        if self._robot.is_connected:
            self._robot.disconnect()


class EmergencyStop:
    def __init__(self, keyboard_enabled: bool, sentinel: Path | None) -> None:
        self.event = threading.Event()
        self.sentinel = sentinel
        if keyboard_enabled and sys.stdin.isatty():
            thread = threading.Thread(target=self._keyboard_worker, daemon=True, name="keyboard-estop")
            thread.start()

    def _keyboard_worker(self) -> None:
        LOG.warning("软件急停已启用：控制期间按 Enter（或 Ctrl+C）立即断开并卸力。")
        line = sys.stdin.readline()
        if line != "":
            self.event.set()

    def is_set(self) -> bool:
        if self.sentinel is not None and self.sentinel.exists():
            LOG.error("检测到急停文件：%s", self.sentinel)
            self.event.set()
        return self.event.is_set()


def verification_signature(args: argparse.Namespace, checkpoint: Path, base_policy: str) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint),
        "base_policy": base_policy,
        "wan_pretrained_path": str(_expand(args.wan_pretrained_path)),
        "attn_mode": "torch",
        "video_steps": args.video_steps,
        "action_steps": args.action_steps,
        "guidance_scale": args.guidance_scale,
        "action_guidance_scale": args.action_guidance_scale,
        "camera_order": list(OBS_CAM_KEYS),
        "used_action_channel_ids": list(USED_ACTION_CHANNEL_IDS),
        "q01": Q01.tolist(),
        "q99": Q99.tolist(),
    }


def validate_live_verification(
    args: argparse.Namespace,
    checkpoint: Path,
    base_policy: str,
) -> None:
    report_path = _expand(args.verification_report)
    if not report_path.is_file():
        raise RuntimeError(
            f"真实执行需要验证报告 {report_path}。先运行同参数的 --mode verify；dry-run 不需要报告。"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = verification_signature(args, checkpoint, base_policy)
    mismatches = {
        key: (report.get("signature", {}).get(key), value)
        for key, value in expected.items()
        if report.get("signature", {}).get(key) != value
    }
    if not report.get("passed", False) or mismatches:
        raise RuntimeError(
            f"验证报告未通过或与本次配置不一致：passed={report.get('passed')} mismatches={mismatches}"
        )


def summarize_predictions(pred_raw: torch.Tensor, gt_raw: torch.Tensor) -> dict[str, Any]:
    mse = ((pred_raw - gt_raw) ** 2).mean(dim=0)
    amplitude = float(pred_raw.std(0).sum() / gt_raw.std(0).sum())
    gripper_min = float(pred_raw[:, -1].min())
    metrics = {
        "mean_mse": float(mse.mean()),
        "amplitude_pct": amplitude * 100.0,
        "gripper_mse": float(mse[-1]),
        "gripper_min": gripper_min,
        "per_dim_mse": {name: float(value) for name, value in zip(ACTION_NAMES, mse.tolist(), strict=True)},
    }
    LOG.info(
        "验证指标：mean_mse=%.2f amplitude=%.1f%% gripper_mse=%.2f gripper_min=%.2f",
        metrics["mean_mse"],
        metrics["amplitude_pct"],
        metrics["gripper_mse"],
        metrics["gripper_min"],
    )
    return metrics


def verify_offline(
    args: argparse.Namespace,
    policy,
    config,
    checkpoint: Path,
    base_policy: str,
) -> bool:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = _expand(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"held-out dataset not found: {dataset_root}. Copy episode {args.episode} before verification."
        )
    LOG.info("解码 held-out episode %d：%s", args.episode, dataset_root)
    raw_ds = LeRobotDataset(
        args.dataset_repo_id,
        root=dataset_root,
        episodes=[args.episode],
    )
    count = min(len(raw_ds), args.max_verify_frames) if args.max_verify_frames else len(raw_ds)
    task = raw_ds[0]["task"]
    policy.reset()
    predictions: list[torch.Tensor] = []
    ground_truth: list[torch.Tensor] = []

    for index in range(count):
        sample = raw_ds[index]
        batch = {key: sample[key].unsqueeze(0).to(config.device) for key in config.obs_cam_keys}
        # CRITICAL: exactly the raw measured state, [1, 6].
        batch["observation.state"] = sample["observation.state"].unsqueeze(0).to(config.device)
        batch["task"] = [task]
        with torch.inference_mode():
            action_norm = policy.select_action(batch)
        predictions.append(action_norm.detach().cpu().reshape(-1))
        ground_truth.append(sample["action"].detach().cpu())
        if (index + 1) % 32 == 0:
            LOG.info("验证进度 %d/%d", index + 1, count)

    pred_raw = denormalize_action(torch.stack(predictions))
    gt_raw = torch.stack(ground_truth)
    metrics = summarize_predictions(pred_raw, gt_raw)
    thresholds = {
        "max_mean_mse": args.max_mean_mse,
        "min_amplitude_pct": args.min_amplitude_pct,
        "max_gripper_mse": args.max_gripper_mse,
        "max_gripper_min": args.max_gripper_min,
    }
    passed = (
        metrics["mean_mse"] <= thresholds["max_mean_mse"]
        and metrics["amplitude_pct"] >= thresholds["min_amplitude_pct"]
        and metrics["gripper_mse"] <= thresholds["max_gripper_mse"]
        and metrics["gripper_min"] <= thresholds["max_gripper_min"]
    )
    report_path = _expand(args.verification_report)
    plot_path = report_path.with_suffix(".png")
    trace_path = report_path.with_suffix(".npz")
    from plotting import plot_action_compare

    plot_action_compare(
        pred_raw,
        gt_raw,
        plot_path,
        title=(
            f"select_action_statefb | ep{args.episode} | "
            f"mse={metrics['mean_mse']:.1f} amp={metrics['amplitude_pct']:.0f}%"
        ),
    )
    np.savez_compressed(
        trace_path,
        prediction=pred_raw.numpy(),
        ground_truth=gt_raw.numpy(),
        action_names=np.asarray(ACTION_NAMES),
    )
    report = {
        "passed": passed,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "driver": "select_action_statefb",
        "plot": str(plot_path),
        "trace": str(trace_path),
        "dataset": {
            "repo_id": args.dataset_repo_id,
            "root": str(dataset_root),
            "episode": args.episode,
            "frames": count,
        },
        "signature": verification_signature(args, checkpoint, base_policy),
        "metrics": metrics,
        "thresholds": thresholds,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("验证%s；报告已写入 %s", "通过" if passed else "失败", report_path)
    return passed


def _resample_or_pad(items: list[Any], target: int) -> list[Any]:
    if not items:
        raise ValueError("cannot build feedback history from an empty list")
    if target <= 0:
        raise ValueError("history target must be positive")
    if len(items) > target:
        indices = torch.linspace(0, len(items) - 1, target).round().long().tolist()
        return [items[index] for index in indices]
    return [*items, *([items[-1]] * (target - len(items)))]


def predict_next_chunk_with_state_feedback(
    policy,
    observation_history: list[dict[str, Any]],
    state_history: list[torch.Tensor],
    executed_action_count: int,
) -> torch.Tensor:
    """PolicyServer-compatible refill with the state-feedback override.

    ``executed_action_count`` is 12 for LingBot-VA's first emitted chunk and 16
    thereafter.  Both image and measured-state histories are aligned to that same
    length.  This preserves the verified sync path's first refill (F=3), rather than
    accidentally padding it to F=4.
    """
    action_per_frame = int(policy.config.action_per_frame)
    if executed_action_count % action_per_frame:
        raise ValueError(
            f"executed_action_count={executed_action_count} must be divisible by "
            f"action_per_frame={action_per_frame}"
        )
    history_capacity = int(policy.observation_history_size)
    if executed_action_count > history_capacity:
        raise ValueError(
            f"executed_action_count={executed_action_count} exceeds policy "
            f"observation_history_size={history_capacity}"
        )
    batches = _resample_or_pad(observation_history, executed_action_count)
    states = _resample_or_pad(state_history, executed_action_count)

    policy.set_observation_history(batches)
    # set_observation_history pads short input to observation_history_size (16).  The
    # first emitted chunk is only 12 actions because conditioning-frame actions are
    # dropped; trim back to 12 to reproduce select_action's N=12 -> F=3 behavior.
    if len(policy._obs_buffer) > executed_action_count:
        policy._obs_buffer = policy._obs_buffer[:executed_action_count]

    executed = policy._states_to_executed(states)
    if executed is None:
        raise RuntimeError("measured-state history did not produce an executed-action tensor")
    expected_frames = executed_action_count // action_per_frame
    if executed.shape[2] != expected_frames:
        raise RuntimeError(
            f"state feedback shape mismatch: got {tuple(executed.shape)}, expected F={expected_frames}"
        )
    policy._executed_actions = executed
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(None)
    return chunk.detach().cpu()


def _target_to_numpy(action_norm: torch.Tensor) -> np.ndarray:
    action = denormalize_action(action_norm.reshape(-1)).detach().float().cpu()
    if action.numel() != 6 or not torch.isfinite(action).all():
        raise RuntimeError(f"invalid policy action: shape={tuple(action.shape)} values={action.tolist()}")
    return action.numpy()


def _loop_should_stop(
    estop: EmergencyStop,
    start_time: float,
    steps: int,
    duration: float,
    max_steps: int | None,
) -> bool:
    return (
        estop.is_set()
        or (duration > 0 and time.monotonic() - start_time >= duration)
        or (max_steps is not None and steps >= max_steps)
    )


def _sleep_to_rate(tick_started: float, fps: float) -> None:
    remaining = 1.0 / fps - (time.monotonic() - tick_started)
    if remaining > 0:
        time.sleep(remaining)


def run_sync(
    args: argparse.Namespace,
    policy,
    config,
    robot: SO101Controller,
    limiter: SafetyLimiter,
    estop: EmergencyStop,
) -> None:
    policy.reset()
    start = time.monotonic()
    steps = 0
    LOG.warning("启动同步控制：execute=%s；每次 chunk refill 会阻塞。", args.execute)

    while not _loop_should_stop(estop, start, steps, args.duration, args.max_steps):
        tick = time.monotonic()
        raw_obs = robot.get_observation()
        state = state_from_raw_observation(raw_obs).squeeze(0).numpy()
        batch = move_batch(build_live_batch(raw_obs, args.task), config.device)
        with torch.inference_mode():
            action_norm = policy.select_action(batch)
        requested = _target_to_numpy(action_norm)
        target = limiter.apply(requested, state)
        if args.execute:
            robot.send_target(target)
        if steps % args.log_every == 0:
            LOG.info(
                "step=%d frame_st_id=%s requested=%s safe=%s",
                steps,
                getattr(policy, "_frame_st_id", None),
                np.round(requested, 2).tolist(),
                np.round(target, 2).tolist(),
            )
        steps += 1
        _sleep_to_rate(tick, args.fps)
    LOG.info("同步控制结束：steps=%d wall=%.1fs", steps, time.monotonic() - start)


def _wait_for_future(
    future: Future[torch.Tensor],
    estop: EmergencyStop,
) -> torch.Tensor:
    while not future.done():
        if estop.is_set():
            future.cancel()
            raise KeyboardInterrupt("emergency stop during inference")
        time.sleep(0.05)
    return future.result()


def run_async(
    args: argparse.Namespace,
    policy,
    config,
    robot: SO101Controller,
    limiter: SafetyLimiter,
    estop: EmergencyStop,
) -> None:
    """Run one KV-cache owner thread while the main thread maintains robot I/O."""
    policy.reset()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lingbot-refill")
    first_raw = robot.get_observation()
    first_batch = move_batch(build_live_batch(first_raw, args.task), config.device)
    LOG.info("异步首块推理开始；首块到达前机器人保持当前位置。")
    first_future = executor.submit(lambda: policy.predict_action_chunk(first_batch).detach().cpu())
    try:
        first_chunk = _wait_for_future(first_future, estop)
    except BaseException:
        first_future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    queue: deque[torch.Tensor] = deque(first_chunk.transpose(0, 1))
    current_chunk_size = len(queue)
    if current_chunk_size <= 0:
        raise RuntimeError("policy returned an empty first chunk")
    LOG.info("首块到达：%d actions（预期 12）", current_chunk_size)

    observation_history: list[dict[str, Any]] = []
    state_history: list[torch.Tensor] = []
    refill_future: Future[torch.Tensor] | None = None
    refill_started = False
    skip_history_once = True
    last_target: np.ndarray | None = None
    start = time.monotonic()
    steps = 0
    holds = 0
    chunk_index = 0

    try:
        while not _loop_should_stop(estop, start, steps, args.duration, args.max_steps):
            tick = time.monotonic()
            raw_obs = robot.get_observation()
            batch_cpu = build_live_batch(raw_obs, args.task)
            measured = batch_cpu["observation.state"].squeeze(0).numpy()

            if refill_future is not None and refill_future.done():
                error = refill_future.exception()
                if error is not None:
                    raise RuntimeError("background LingBot-VA refill failed") from error
                if not queue:
                    next_chunk = refill_future.result()
                    queue.extend(next_chunk.transpose(0, 1))
                    current_chunk_size = len(queue)
                    chunk_index += 1
                    LOG.info(
                        "chunk %d 到达：%d actions；此前 hold=%d ticks",
                        chunk_index,
                        current_chunk_size,
                        holds,
                    )
                    observation_history.clear()
                    state_history.clear()
                    refill_future = None
                    refill_started = False
                    skip_history_once = True
                    holds = 0

            # The first observation of a newly arrived chunk is its conditioning
            # state, not the result of one of that chunk's actions.
            if skip_history_once and queue:
                skip_history_once = False
            elif not refill_started:
                observation_history.append(batch_cpu)
                state_history.append(batch_cpu["observation.state"].detach())

            margin = min(args.async_prefetch_margin, max(0, current_chunk_size - 4))
            observations_needed = current_chunk_size - margin
            if not refill_started and len(observation_history) >= observations_needed:
                if margin:
                    LOG.warning(
                        "实验性提前预取：仅观测 %d/%d 个动作结果；余下历史将用最后实测帧/状态补齐。"
                        "修改该值后必须重新做闭环验证。",
                        len(observation_history),
                        current_chunk_size,
                    )
                history_snapshot = list(observation_history)
                states_snapshot = list(state_history)
                refill_future = executor.submit(
                    predict_next_chunk_with_state_feedback,
                    policy,
                    history_snapshot,
                    states_snapshot,
                    current_chunk_size,
                )
                refill_started = True
                LOG.info(
                    "启动 chunk %d refill：measured_history=%d target=%d queue_remaining=%d",
                    chunk_index + 1,
                    len(states_snapshot),
                    current_chunk_size,
                    len(queue),
                )

            if queue:
                action_norm = queue.popleft()
                requested = _target_to_numpy(action_norm)
                last_target = limiter.apply(requested, measured)
                if args.execute:
                    robot.send_target(last_target)
                if steps % args.log_every == 0:
                    LOG.info(
                        "step=%d chunk=%d q=%d requested=%s safe=%s",
                        steps,
                        chunk_index,
                        len(queue),
                        np.round(requested, 2).tolist(),
                        np.round(last_target, 2).tolist(),
                    )
                steps += 1
            else:
                # Re-send the already safety-limited target.  This is especially
                # important for sustaining a stalled gripper close.
                if args.execute and last_target is not None:
                    robot.send_target(last_target)
                holds += 1

            _sleep_to_rate(tick, args.fps)
    finally:
        if refill_future is not None:
            refill_future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
    LOG.info(
        "异步控制结束：executed_steps=%d chunks=%d wall=%.1fs",
        steps,
        chunk_index + 1,
        time.monotonic() - start,
    )


def prepare_official_async_checkpoint(
    args: argparse.Namespace,
    lora_checkpoint: Path,
    base_policy: str,
    wan_root: Path,
) -> Path:
    """Package the adapter, config, and processors expected by PolicyServer."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE

    output = _expand(args.async_checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = build_policy_config(args, base_policy, wan_root)
    config.use_peft = True
    config.input_features = {
        key: PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, args.camera_height, args.camera_width),
        )
        for key in OBS_CAM_KEYS
    }
    config.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(6,))
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(len(USED_ACTION_CHANNEL_IDS),))
    }
    config._save_pretrained(output)

    config_path = output / "config.json"
    config_json = json.loads(config_path.read_text(encoding="utf-8"))
    config_json["use_peft"] = True
    config_path.write_text(json.dumps(config_json, indent=2) + "\n", encoding="utf-8")

    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        source = lora_checkpoint / filename
        destination = output / filename
        if not destination.exists() or source.stat().st_size != destination.stat().st_size:
            shutil.copy2(source, destination)

    adapter_config_path = output / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    adapter_config["base_model_name_or_path"] = base_policy
    adapter_config_path.write_text(
        json.dumps(adapter_config, indent=2) + "\n",
        encoding="utf-8",
    )

    stats = {"action": {"q01": Q01.clone(), "q99": Q99.clone()}}
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(output)
    postprocessor.save_pretrained(output)
    LOG.info(
        "官方异步 checkpoint 已就绪：%s（torch attention，去噪=%d/%d）",
        output,
        args.video_steps,
        args.action_steps,
    )
    return output


def _clear_server_observation_history(server) -> None:
    with server._observation_history_lock:
        server._observation_history = []


@dataclass
class OfficialRefillBoundary:
    """Decide when continuous observations should trigger an official-server refill."""

    margin: int
    armed: bool = False

    def update(
        self,
        queue_size: int | float | None,
        must_go: bool,
        has_processed_observation: bool,
    ) -> tuple[bool, bool]:
        """Return ``(start_execution_history_here, enqueue_for_inference)``."""
        if not has_processed_observation:
            self.armed = False
            return False, True
        if not self.armed and queue_size is not None and queue_size > self.margin:
            self.armed = True
            return True, False
        eligible = must_go or (self.armed and queue_size is not None and queue_size <= self.margin)
        if eligible:
            self.armed = False
        return False, eligible


def make_lingbot_policy_server(args: argparse.Namespace):
    """Create a PolicyServer subclass with exact SO-101 state/KV feedback."""
    from lerobot.async_inference.helpers import raw_observation_to_observation
    from lerobot.async_inference.policy_server import PolicyServer

    class LingBotStateFeedbackPolicyServer(PolicyServer):
        def __init__(self, config) -> None:
            super().__init__(config)
            self._previous_emitted_actions: int | None = None
            self._refill_boundary = OfficialRefillBoundary(args.async_prefetch_margin)

        def _reset_server(self) -> None:
            super()._reset_server()
            self._previous_emitted_actions = None
            self._refill_boundary = OfficialRefillBoundary(args.async_prefetch_margin)

        def _load_policy(self, policy_specs):
            loaded = super()._load_policy(policy_specs)
            policy = loaded.get_base_model() if hasattr(loaded, "get_base_model") else loaded

            if str(EXAMPLE_DIR) not in sys.path:
                sys.path.insert(0, str(EXAMPLE_DIR))
            from common import attach_text_embed_cache
            from state_feedback import attach_state_feedback

            attach_text_embed_cache(policy)
            attach_state_feedback(policy, Q01, Q99, verbose=True)
            policy.eval()
            if not hasattr(policy, "_states_to_executed"):
                raise RuntimeError("官方 PolicyServer 未能安装 state-feedback patch")

            # SendPolicyInstructions blocks until this finishes. The one constant task is
            # cached once, so no later chunk pays the CPU UMT5 cost.
            self.logger.info("Warming LingBot-VA frozen modules and task embedding")
            policy._ensure_frozen_modules()
            policy._maybe_init_prompt({"task": [policy_specs.task]})
            policy.reset()
            self.logger.info("LingBot-VA state feedback and text cache are ready")
            return policy

        def _enqueue_observation(self, obs) -> bool:
            """Record every frame, but request one inference at the configured boundary."""
            metadata = obs.get_metadata() if hasattr(obs, "get_metadata") else {}
            queue_size = metadata.get("queue_size_at_capture") if isinstance(metadata, dict) else None
            begin_history, eligible = self._refill_boundary.update(
                queue_size,
                bool(obs.must_go),
                self.last_processed_obs is not None,
            )

            # Seeing a non-low queue means the new chunk reached the client. Drop the
            # conditioning/idle frames accumulated before its first action was executed.
            # RobotClient performs the action before capturing this observation, so keep
            # the current frame: it is already the result of action 0.
            if begin_history:
                with self._observation_history_lock:
                    self._observation_history = [obs]
                return False

            if not eligible:
                return False
            return super()._enqueue_observation(obs)

        def _push_observation_history(self) -> None:
            target = self._previous_emitted_actions
            with self._observation_history_lock:
                history, self._observation_history = self._observation_history, []

            # First observation is only the first chunk's conditioning frame.
            if target is None:
                return
            if not history:
                raise RuntimeError("下一 chunk 缺少上一 chunk 的实测观测历史")

            selected = _resample_or_pad(history, target)
            batches = [
                self.preprocessor(
                    raw_observation_to_observation(
                        timed_observation.get_observation(),
                        self.lerobot_features,
                        self.policy_image_features,
                    )
                )
                for timed_observation in selected
            ]
            states = [batch["observation.state"].detach() for batch in batches]
            self.policy.set_observation_history(batches)
            if len(self.policy._obs_buffer) > target:
                self.policy._obs_buffer = self.policy._obs_buffer[:target]

            executed = self.policy._states_to_executed(states)
            if executed is None:
                raise RuntimeError(f"{len(states)} 个实测状态不足以生成 state feedback")
            expected_frames = target // int(self.policy.config.action_per_frame)
            if executed.shape[2] != expected_frames:
                raise RuntimeError(f"state feedback shape={tuple(executed.shape)}，预期 F={expected_frames}")
            self.policy._executed_actions = executed
            self.logger.info(
                "LingBot-VA state feedback: measured=%d target=%d -> F=%d",
                len(history),
                target,
                expected_frames,
            )

        def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
            chunk = super()._get_action_chunk(observation)
            self._previous_emitted_actions = int(chunk.shape[1])
            # Frames arriving during the slow inference describe a held robot, not the
            # execution of the chunk that has only just been emitted.
            _clear_server_observation_history(self)
            return chunk

    from lerobot.async_inference.configs import PolicyServerConfig

    server_config = PolicyServerConfig(
        host=args.server_host,
        port=args.server_port,
        fps=max(1, round(args.fps)),
        inference_latency=0.0,
        obs_queue_timeout=2.0,
        record_timeline=True,
        timeline_log_dir=args.timeline_log_dir,
    )
    return LingBotStateFeedbackPolicyServer(server_config), server_config


def run_official_server(args: argparse.Namespace) -> None:
    """Serve LingBot-VA through LeRobot's official gRPC async transport."""
    import grpc

    from lerobot.transport import services_pb2_grpc

    policy_server, config = make_lingbot_policy_server(args)
    grpc_server = grpc.server(ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, grpc_server)
    bound_port = grpc_server.add_insecure_port(f"{config.host}:{config.port}")
    if bound_port == 0:
        raise RuntimeError(f"无法监听 {config.host}:{config.port}")
    grpc_server.start()
    LOG.warning(
        "官方 PolicyServer 已启动：%s:%d，prefetch_margin=%d；Ctrl+C 停止",
        config.host,
        bound_port,
        args.async_prefetch_margin,
    )
    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        LOG.info("收到 Ctrl+C，停止 PolicyServer")
    finally:
        policy_server.stop()
        grpc_server.stop(grace=0)


def build_official_robot_config(args: argparse.Namespace):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SOFollowerRobotConfig

    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=args.front_camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            color_mode="rgb",
            fourcc=args.camera_fourcc,
        ),
        "right": OpenCVCameraConfig(
            index_or_path=args.right_camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            color_mode="rgb",
            fourcc=args.camera_fourcc,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            color_mode="rgb",
            fourcc=args.wrist_fourcc,
        ),
    }
    return SOFollowerRobotConfig(
        id=args.robot_id,
        port=args.robot_port,
        cameras=cameras,
        use_degrees=True,
        max_relative_target=None,
    )


def run_official_client(args: argparse.Namespace, async_checkpoint: Path) -> None:
    """Run the official RobotClient with continuous history streaming and safety."""
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient

    config = RobotClientConfig(
        policy_type="lingbot_va",
        pretrained_name_or_path=str(async_checkpoint),
        robot=build_official_robot_config(args),
        actions_per_chunk=16,
        task=args.task,
        server_address=f"{args.server_host}:{args.server_port}",
        policy_device=args.device,
        client_device="cpu",
        # The stock client conflates observation streaming and replan timing. We
        # override the former below so every tick is streamed; the custom server
        # decides when the queue reaches --async-prefetch-margin.
        chunk_size_threshold=1.0,
        fps=max(1, round(args.fps)),
        aggregate_fn_name="latest_only",
        run_seconds=args.duration if args.duration > 0 else None,
        record_timeline=True,
        timeline_log_dir=args.timeline_log_dir,
    )
    client = RobotClient(config)
    client.robot.bus.write("Max_Torque_Limit", "gripper", args.gripper_max_torque, num_retry=3)
    client.robot.bus.write(
        "Protection_Current",
        "gripper",
        args.gripper_protection_current,
        num_retry=3,
    )
    client.robot.bus.write(
        "Overload_Torque",
        "gripper",
        args.gripper_overload_torque,
        num_retry=3,
    )
    estop = EmergencyStop(
        not args.no_keyboard_estop,
        _expand(args.estop_file) if args.estop_file else None,
    )
    client.robot = SafeRobotProxy(
        client.robot,
        make_limiter(args),
        args.execute,
        estop,
        args.log_every,
    )
    # Stream every observation. Only the custom server's queue-boundary state
    # machine is allowed to trigger a prediction.
    client._ready_to_send_observation = lambda: True

    receiver: threading.Thread | None = None
    try:
        if not client.start():
            raise RuntimeError(f"无法连接 PolicyServer {config.server_address}")
        LOG.warning(
            "官方异步客户端启动：%s，execute=%s，fps=%d，duration=%s",
            config.server_address,
            args.execute,
            config.fps,
            args.duration,
        )
        receiver = threading.Thread(
            target=client.receive_actions,
            kwargs={"verbose": True},
            daemon=True,
            name="official-action-receiver",
        )
        receiver.start()
        client.control_loop(task=args.task, verbose=False)
    except KeyboardInterrupt:
        LOG.error("急停/中断：停止官方异步客户端")
    finally:
        client.stop()
        if receiver is not None:
            receiver.join(timeout=5)
        LOG.warning("官方异步客户端已断开机器人并停止")


def make_limiter(args: argparse.Namespace) -> SafetyLimiter:
    return SafetyLimiter(
        joint_min=args.joint_min.copy(),
        joint_max=args.joint_max.copy(),
        max_arm_step=args.max_arm_step_deg,
        max_gripper_command_step=args.max_gripper_command_step,
    )


def default_base_policy() -> str:
    local = Path.home() / "Projects/models/lingbot_va_base_lerobot"
    return str(local) if (local / "model.safetensors").is_file() else "lerobot/lingbot_va_base"


def default_dataset_root() -> str:
    thor_copy = Path.home() / "Datasets/three_cubes_1"
    return str(thor_copy) if thor_copy.exists() else "/data/rxhuang/three_cubes_1"


def default_verification_report(video_steps: int, action_steps: int) -> str:
    return f"outputs/thor_lingbot_va_step15000_v{video_steps}_a{action_steps}_verification.json"


def default_async_checkpoint_dir(video_steps: int, action_steps: int) -> Path:
    return Path.home() / "Projects" / "models" / f"lingbot_va_async_v{video_steps}_a{action_steps}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Thor 上部署 state-conditioned LingBot-VA 到 SO-101",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=(
            "preflight",
            "verify",
            "sync",
            "async",
            "official-server",
            "official-client",
        ),
        default="preflight",
    )
    parser.add_argument(
        "--lora-checkpoint",
        default=str(Path.home() / "Projects/models/lingbot_va_lora"),
        help="step_15000 adapter 目录，或包含它的父目录",
    )
    parser.add_argument(
        "--base-policy-path",
        default=default_base_policy(),
        help="LeRobot 格式 base（根目录含 config.json + model.safetensors），也可用 HF repo id",
    )
    parser.add_argument(
        "--wan-pretrained-path",
        default=str(Path.home() / "Projects/models/lingbot-va-base"),
        help="冻结 Wan 资产根目录（含 vae/text_encoder/tokenizer）",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-encoder-device", default="cpu")
    parser.add_argument("--video-steps", type=int, default=20)
    parser.add_argument("--action-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--action-guidance-scale", type=float, default=1.0)

    parser.add_argument("--dataset-root", default=default_dataset_root())
    parser.add_argument("--dataset-repo-id", default="three_cubes_1")
    parser.add_argument("--episode", type=int, default=95)
    parser.add_argument("--max-verify-frames", type=int, default=320)
    parser.add_argument(
        "--verification-report",
        default=None,
        help="验证 JSON；默认按去噪配置自动生成 ..._v<video>_a<action>_verification.json",
    )
    parser.add_argument("--max-mean-mse", type=float, default=150.0)
    parser.add_argument("--min-amplitude-pct", type=float, default=70.0)
    parser.add_argument("--max-gripper-mse", type=float, default=75.0)
    parser.add_argument("--max-gripper-min", type=float, default=7.5)

    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="follower_arm")
    parser.add_argument("--front-camera", type=int, default=2)
    parser.add_argument("--right-camera", type=int, default=0)
    parser.add_argument("--wrist-camera", type=int, default=4)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-fourcc", default="MJPG")
    parser.add_argument("--wrist-fourcc", default="YUYV")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际发送动作；未指定时只读机器人/相机并打印安全限幅后的目标",
    )

    parser.add_argument(
        "--joint-min",
        type=parse_six_floats,
        default=DEFAULT_JOINT_MIN,
        help="六维绝对部署下限，逗号分隔",
    )
    parser.add_argument(
        "--joint-max",
        type=parse_six_floats,
        default=DEFAULT_JOINT_MAX,
        help="六维绝对部署上限，逗号分隔",
    )
    parser.add_argument("--max-arm-step-deg", type=float, default=2.0)
    parser.add_argument("--max-gripper-command-step", type=float, default=4.0)
    parser.add_argument("--gripper-max-torque", type=int, default=500)
    parser.add_argument("--gripper-protection-current", type=int, default=250)
    parser.add_argument("--gripper-overload-torque", type=int, default=25)
    parser.add_argument("--no-keyboard-estop", action="store_true")
    parser.add_argument("--estop-file", type=Path)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8080)
    parser.add_argument(
        "--async-checkpoint-dir",
        type=Path,
        default=None,
        help="官方 PolicyServer 使用的完整 PEFT+processor checkpoint；默认按去噪配置生成",
    )
    parser.add_argument("--timeline-log-dir", default="logs/lingbot_va_official_async")
    parser.add_argument(
        "--async-prefetch-margin",
        type=int,
        default=0,
        help="还剩多少动作时提前 refill；0=完整实测历史（正确性默认），>0 会补齐未知未来历史，须重验",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0 or args.camera_fps <= 0:
        raise ValueError("fps values must be positive")
    if args.video_steps <= 0 or args.action_steps <= 0:
        raise ValueError("denoising steps must be positive")
    if args.async_prefetch_margin < 0:
        raise ValueError("--async-prefetch-margin must be >= 0")
    if not 0 <= args.gripper_max_torque <= 1000:
        raise ValueError("--gripper-max-torque must be in [0,1000]")
    if not 0 <= args.gripper_protection_current <= 500:
        raise ValueError("--gripper-protection-current must be in [0,500]")
    if not 0 <= args.gripper_overload_torque <= 100:
        raise ValueError("--gripper-overload-torque must be in [0,100]")
    if not 1 <= args.server_port <= 65535:
        raise ValueError("--server-port must be in [1,65535]")
    if args.execute and args.mode not in ("sync", "async", "official-client"):
        raise ValueError("--execute is only valid with --mode sync/async/official-client")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = build_parser().parse_args()
    if args.verification_report is None:
        args.verification_report = default_verification_report(args.video_steps, args.action_steps)
    if args.async_checkpoint_dir is None:
        args.async_checkpoint_dir = default_async_checkpoint_dir(
            args.video_steps,
            args.action_steps,
        )
    validate_args(args)

    checkpoint = resolve_lora_checkpoint(args.lora_checkpoint)
    base_policy = resolve_base_policy(args.base_policy_path)
    wan_root = validate_wan_assets(args.wan_pretrained_path)
    LOG.info("LoRA=%s", checkpoint)
    LOG.info("LeRobot base=%s", base_policy)
    LOG.info("Wan frozen assets=%s", wan_root)
    if base_policy == str(wan_root):
        raise RuntimeError(
            "--base-policy-path and --wan-pretrained-path have different formats/roles and cannot be identical"
        )
    if args.mode == "preflight":
        LOG.info("资产预检通过。下一步运行 --mode verify；此模式未加载权重、未连接机器人。")
        return 0

    if args.execute:
        validate_live_verification(args, checkpoint, base_policy)
        LOG.warning("验证门禁通过；本次会向真实电机发送目标。")

    if args.mode in ("official-server", "official-client"):
        async_checkpoint = prepare_official_async_checkpoint(
            args,
            checkpoint,
            base_policy,
            wan_root,
        )
        if args.mode == "official-server":
            run_official_server(args)
        else:
            run_official_client(args, async_checkpoint)
        return 0

    policy, config = load_policy(args, checkpoint, base_policy, wan_root)
    if args.mode == "verify":
        return 0 if verify_offline(args, policy, config, checkpoint, base_policy) else 2

    warm_text_cache_before_robot_connect(policy, args.task)
    robot = SO101Controller(args)
    limiter = make_limiter(args)
    try:
        robot.connect(args)
        # Start stdin monitoring only after any interactive calibration prompt has
        # completed, otherwise the two readers would race for the same terminal.
        estop = EmergencyStop(
            not args.no_keyboard_estop,
            _expand(args.estop_file) if args.estop_file else None,
        )
        LOG.warning(
            "%s：camera order=[front,right,wrist]；joint envelope min=%s max=%s",
            "LIVE" if args.execute else "DRY-RUN（绝不发送动作）",
            args.joint_min.tolist(),
            args.joint_max.tolist(),
        )
        if args.mode == "sync":
            run_sync(args, policy, config, robot, limiter, estop)
        else:
            run_async(args, policy, config, robot, limiter, estop)
    except KeyboardInterrupt:
        LOG.error("急停/中断：立即退出控制循环。")
    finally:
        robot.disconnect()
        LOG.warning("机器人已断开，默认配置下扭矩已卸载。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
