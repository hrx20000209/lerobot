import sys
import types

import pytest
import torch
from torch import nn

from lerobot.policies.lingbo_va.eval_utils import _override_attn_mode, _set_attn_op


def _custom_sdpa(q, k, v):
    return q


@pytest.fixture(autouse=True)
def _fake_wan_va_modules_model(monkeypatch):
    """`_override_attn_mode` imports `wan_va.modules.model.custom_sdpa`/`flash_attn_func`, which
    requires the private vendored `lingbot-va` checkout to be on sys.path (normally done by
    `policy._install_lingbo_import_path()`). Stub the full package chain so this test is
    self-contained and doesn't depend on that checkout existing in CI."""
    fake_model = types.ModuleType("wan_va.modules.model")
    fake_model.custom_sdpa = _custom_sdpa
    fake_model.flash_attn_func = None
    fake_modules_pkg = types.ModuleType("wan_va.modules")
    fake_wan_va_pkg = types.ModuleType("wan_va")
    monkeypatch.setitem(sys.modules, "wan_va", fake_wan_va_pkg)
    monkeypatch.setitem(sys.modules, "wan_va.modules", fake_modules_pkg)
    monkeypatch.setitem(sys.modules, "wan_va.modules.model", fake_model)


class _FlexAttnFunc(nn.Module):
    """Mimics the real vendored FlexAttnFunc: an nn.Module with no trainable parameters,
    assigned to `attn_op` for attn_mode="flex" (see wan_va/modules/model.py)."""

    def forward(self, q, k, v):
        return q


class _FakeWanAttention(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.to_q = nn.Linear(2, 2)
        self.attn_op = _FlexAttnFunc() if mode == "flex" else _custom_sdpa


def test_set_attn_op_unregisters_flex_submodule_before_assigning_plain_function() -> None:
    # Regression test: WanAttention.attn_op is an nn.Module (FlexAttnFunc) when attn_mode="flex",
    # so plain `module.attn_op = custom_sdpa` raises
    # "TypeError: cannot assign ... as child module 'attn_op' (torch.nn.Module or None expected)".
    attn = _FakeWanAttention("flex")
    assert "attn_op" in attn._modules

    _set_attn_op(attn, _custom_sdpa)

    assert attn.attn_op is _custom_sdpa
    assert "attn_op" not in attn._modules


def test_attn_op_override_and_restore_round_trips_through_module_registration() -> None:
    transformer = nn.Module()
    transformer.blocks = nn.ModuleList([_FakeWanAttention("flex"), _FakeWanAttention("flex")])
    original_ops = [block.attn_op for block in transformer.blocks]

    _override_attn_mode(transformer, "torch")
    for block in transformer.blocks:
        assert block.attn_op is not None
        assert "attn_op" not in block._modules
        assert callable(block.attn_op)

    # Restore exactly like build_weight_sharing_server's context-manager exit does: plain
    # assignment, which PyTorch's nn.Module.__setattr__ handles correctly regardless of whether
    # the name currently lives in __dict__ (post-override) or _modules.
    for block, original_op in zip(transformer.blocks, original_ops, strict=False):
        block.attn_op = original_op

    for block, original_op in zip(transformer.blocks, original_ops, strict=False):
        assert block.attn_op is original_op
        assert "attn_op" in block._modules
    assert any(isinstance(m, _FlexAttnFunc) for m in transformer.modules())


def test_override_attn_mode_rejects_flex() -> None:
    transformer = nn.Module()
    transformer.blocks = nn.ModuleList([_FakeWanAttention("flex")])
    try:
        _override_attn_mode(transformer, "flex")
        raise AssertionError("expected ValueError for attn_mode='flex'")
    except ValueError:
        pass


def test_override_attn_mode_raises_if_no_attn_op_found() -> None:
    transformer = nn.Module()
    transformer.harmless = nn.Linear(2, 2)
    try:
        _override_attn_mode(transformer, "torch")
        raise AssertionError("expected RuntimeError when no attn_op submodules exist")
    except RuntimeError:
        pass


def test_forward_still_works_after_attn_op_swap() -> None:
    # Sanity: swapped-in plain function is actually callable end-to-end, not just type-compatible.
    attn = _FakeWanAttention("flex")
    _set_attn_op(attn, _custom_sdpa)
    q = torch.randn(1, 2)
    out = attn.attn_op(q, q, q)
    assert torch.equal(out, q)
