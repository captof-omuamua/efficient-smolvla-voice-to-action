"""Diagnostic: the PR's test_smolvla_tuple_kv_parity, with only its DynamicCache crop call changed.

transformers 5.22 no longer accepts crop(<positive absolute length>); the library code already crops with a
negative count (prefix_len - current length). This runs the same comparison with that call, on CPU.
"""
import torch

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from tests.policies.smolvla.test_smolvla_compile_parity import _build_tiny_smolvla_policy_and_batch

policy, batch, noise = _build_tiny_smolvla_policy_and_batch(device="cpu")
config = policy.config
with torch.no_grad():
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
        images, img_masks, batch["observation.language.tokens"], batch["observation.language.attention_mask"],
        state=state)
    _, dynamic_kv = policy.model.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(prefix_pad_masks, prefix_att_masks),
        position_ids=torch.cumsum(prefix_pad_masks, dim=1) - 1,
        past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True)
    tuple_kv = tuple((layer.keys.clone(), layer.values.clone()) for layer in dynamic_kv.layers)
    snapshot = tuple((k.clone(), v.clone()) for k, v in tuple_kv)
    prefix_len = prefix_pad_masks.shape[1]
    x_dyn, x_tup = noise.clone(), noise.clone()
    dt = -1.0 / config.num_steps
    for step in range(config.num_steps):
        t = torch.tensor(1.0 + step * dt, dtype=torch.float32).expand(1)
        dynamic_kv.crop(prefix_len - dynamic_kv.get_seq_length())  # the only change: negative count
        v_dyn = policy.model.denoise_step(prefix_pad_masks, dynamic_kv, x_dyn, t)
        v_tup = policy.model.denoise_step(prefix_pad_masks, tuple_kv, x_tup, t)
        assert torch.equal(v_dyn, v_tup), f"step {step}: velocity mismatch"
        x_dyn, x_tup = x_dyn + dt * v_dyn, x_tup + dt * v_tup
    assert torch.equal(x_dyn, x_tup), "final action mismatch"
    for (k, v), (k0, v0) in zip(tuple_kv, snapshot, strict=True):
        assert torch.equal(k, k0) and torch.equal(v, v0), "tuple KV prefix was mutated"
print(f"PASS: DynamicCache and tuple KV give bit-identical velocities for {config.num_steps} steps; prefix unchanged")
