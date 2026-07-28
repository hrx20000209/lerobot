"""Inference-side fix for LingBot-VA closed-loop action drift on SO-101.

Diagnosis (docs/closed_loop_verification.md): `select_action()` feeds the model's OWN
predicted actions back into the KV cache as the executed-action history. Those small errors
compound and the rollout collapses to ~30% motion amplitude. Feeding the MEASURED joint state
back instead recovers ~80% amplitude at the *same* checkpoint (state[t+1] ~= commanded
action[t] on a position-controlled arm, r=0.93-0.997) -- and a real robot always has it.

`attach_state_feedback(policy, q01, q99)` monkeypatches `select_action` (same instance-level
style as `attach_text_embed_cache`, so core lerobot is untouched) to:
  - buffer `batch["observation.state"]` in lockstep with the obs keyframes, and
  - overwrite `policy._executed_actions` from those measured states right before each refill.

The deploy loop must therefore include `observation.state` in every `select_action` batch.
"""

import types

import torch

from common import normalize_action


def attach_state_feedback(policy, q01, q99, verbose: bool = False, hybrid_gripper_channel: int | None = None):
    """Feed measured proprioception back as the executed-action history in the KV cache.

    ``hybrid_gripper_channel``: if set (e.g. 28), that channel is fed the model's OWN commanded
    action instead of the measured state. Rationale (docs/closed_loop_verification.md): on this
    SO-101 the gripper STALLS on the cube during a grasp -- commanded action ~4 but measured state
    ~15 (|a-s|~10.6 during grasp vs ~0.2 elsewhere). Feeding the stalled state tells the model the
    gripper is at 15 and it never grips; feeding the model's own command lets it sustain the close.
    Arm channels (where action~=state, r=0.93-0.997) keep the drift-free measured-state feedback.
    """
    used_ids = list(policy.config.used_action_channel_ids)
    action_dim = policy.config.action_dim
    apf = policy.config.action_per_frame
    q01 = torch.as_tensor(q01, dtype=torch.float32)
    q99 = torch.as_tensor(q99, dtype=torch.float32)
    dbg = {"n": 0}

    def _states_to_executed(self, states):
        """[N measured states] -> model-normalized executed-action tensor [1, action_dim, F, apf, 1]."""
        S = torch.stack([s.reshape(-1) for s in states]).to(self.config.device, torch.float32)  # [N, n_used]
        N = S.shape[0]
        f = N // apf
        if f == 0:
            return None
        S = S[: f * apf]
        S_norm = normalize_action(S, q01, q99)  # [-1, 1], same quantiles as training actions
        full = S.new_zeros(f * apf, action_dim)
        full[:, torch.as_tensor(used_ids, device=S.device)] = S_norm
        full = full.view(f, apf, action_dim).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)  # [1,A,f,apf,1]
        gripper_src = "state"
        if hybrid_gripper_channel is not None:
            prev = getattr(self, "_executed_actions", None)  # model's own commanded chunk [1,A,F,apf,1]
            if prev is not None and prev.shape[2] == full.shape[2]:
                full[:, hybrid_gripper_channel] = prev[:, hybrid_gripper_channel].to(full)
                gripper_src = "model_action"
        if verbose and dbg["n"] < 3:
            print(f"  [state_fb] refill: N_states={N} -> f={f} exec_actions shape={tuple(full.shape)} "
                  f"norm_range=[{S_norm.min():.2f},{S_norm.max():.2f}] gripper_src={gripper_src}", flush=True)
            dbg["n"] += 1
        return full.to(self.dtype)

    def select_action(self, batch, **kwargs):
        self.eval()
        self._ensure_frozen_modules()
        self._maybe_init_prompt(batch)

        if not self._started:
            self._started = True
            self._sf_state_buffer = []
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions.transpose(0, 1))
            self._obs_buffer = []
            self._exec_step = 0
        else:
            if not hasattr(self, "_sf_state_buffer"):
                self._sf_state_buffer = []
            if (self._prev_j + 1) % self._keyframe_stride == 0:
                self._obs_buffer.append(self._extract_raw_obs(batch))
                self._sf_state_buffer.append(batch["observation.state"].detach())
            if len(self._action_queue) == 0:
                # STATE FEEDBACK: replace the model's own predicted history with measured state.
                if self._sf_state_buffer:
                    exec_from_state = self._states_to_executed(self._sf_state_buffer)
                    if exec_from_state is not None:
                        self._executed_actions = exec_from_state
                self._sf_state_buffer = []
                actions = self.predict_action_chunk(None)
                self._action_queue.extend(actions.transpose(0, 1))
                self._exec_step = 0

        self._prev_j = self._exec_step % self.config.action_per_frame
        self._exec_step += 1
        return self._action_queue.popleft()

    policy._states_to_executed = types.MethodType(_states_to_executed, policy)
    policy.select_action = types.MethodType(select_action, policy)
    return policy
