"""Low-VRAM (16GB) pipeline for Ideogram 4.0.

The official ``Ideogram4Pipeline`` holds the text encoder, both transformers and the
VAE on CUDA at once (~15 GiB of weights alone) and builds a *dense* ``llm_features``
tensor of shape ``(B, text+image_tokens, 53248)`` plus an equally large zero tensor
for the negative branch — which overflows 16 GB at >=1024px.

This module re-implements the same maths as the official pipeline (verbatim-faithful
to ``ideogram4.pipeline_ideogram4`` / ``ideogram4.modeling_ideogram4``) with two
changes, WITHOUT editing the installed package:

1. **Staged component loading** — text encoder, then the two transformers, then the
   VAE are each loaded to CUDA, used, and freed before the next phase, so they are
   never resident together.
2. **Sparse conditioning** — the text encoder runs on text tokens only, and the
   conditional/unconditional transformer forwards are re-implemented so the giant
   per-image-token ``llm_features`` (and the negative all-zero tensor) are never
   allocated. This is mathematically identical: the official forward masks
   ``llm_features`` to text positions anyway, and the negative branch is image-only.

Assumes batch size 1 (``process_images_ideogram4`` already calls the pipeline once
per image), so text tokens occupy the front of the sequence with no left padding.

Only imported when "Low VRAM mode: 16GB" is selected, so importing torch / the
official package here is fine.
"""

import gc
import logging
import time

import torch
import torch.nn.functional as F

logger = logging.getLogger("ideogram4")


def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _vram(tag: str):
    if not torch.cuda.is_available():
        return
    try:
        allocated = torch.cuda.memory_allocated() / 2**30
        reserved = torch.cuda.memory_reserved() / 2**30
        peak = torch.cuda.max_memory_allocated() / 2**30
        free, total = torch.cuda.mem_get_info()
        logger.info(
            "Ideogram 4.0 low VRAM 16GB: %s | allocated=%.2f reserved=%.2f peak=%.2f free=%.2f/%.2f GiB",
            tag, allocated, reserved, peak, free / 2**30, total / 2**30,
        )
    except Exception:
        logger.debug("VRAM log failed for %s", tag, exc_info=True)


# ---------------------------------------------------------------------------
# Re-implemented transformer forwards (faithful to Ideogram4Transformer.forward,
# but avoiding the dense per-image-token llm_features tensors).
# ---------------------------------------------------------------------------

def _param_dtype(transformer):
    return getattr(transformer.input_proj, "compute_dtype", None) or transformer.input_proj.weight.dtype


def _conditional_forward_sparse(transformer, *, llm_features_text, x, t, position_ids, segment_ids, indicator, text_token_count):
    """Ideogram4Transformer.forward with text-only llm_features (B, T, llm_dim).

    Equivalent to the official forward: the official zeroes llm_features outside the
    LLM (text) positions, so only the text-position projections contribute. With
    batch size 1 the text tokens are the leading ``text_token_count`` positions.
    """
    from ideogram4.constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR

    dtype = _param_dtype(transformer)
    x = x.to(dtype)
    t = t.to(dtype)
    llm_features_text = llm_features_text.to(dtype)

    indicator = indicator.to(torch.long)
    llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
    output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)

    x = x * output_image_mask
    x = transformer.input_proj(x) * output_image_mask  # (B, L, emb)

    t_cond = transformer.t_embedding(t)
    if t.dim() == 1:
        t_cond = t_cond.unsqueeze(1)
    adaln_input = F.silu(transformer.adaln_proj(t_cond))

    # Project text features only, then place at the leading text positions.
    lf = transformer.llm_cond_norm(llm_features_text)
    lf = transformer.llm_cond_proj(lf)  # (B, T, emb)
    lf = lf * llm_token_mask[:, :text_token_count]

    h = x
    h[:, :text_token_count] = h[:, :text_token_count] + lf

    image_indicator_embedding = transformer.embed_image_indicator(
        (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long)
    )
    h = h + image_indicator_embedding

    cos, sin = transformer.rotary_emb(position_ids)
    cos = cos.to(h.dtype)
    sin = sin.to(h.dtype)

    for layer in transformer.layers:
        h = layer(h, segment_ids=segment_ids, cos=cos, sin=sin, adaln_input=adaln_input)

    out = transformer.final_layer(h, c=adaln_input)
    return out.to(torch.float32)


def _unconditional_forward_image_only(transformer, *, x, t, position_ids, segment_ids, indicator):
    """Ideogram4Transformer.forward for the image-only negative branch.

    The official negative branch passes an all-zero ``llm_features`` and an indicator
    with no LLM positions, so the llm contribution is exactly zero — we skip it (and
    never allocate the big zero tensor).
    """
    from ideogram4.constants import OUTPUT_IMAGE_INDICATOR

    dtype = _param_dtype(transformer)
    x = x.to(dtype)
    t = t.to(dtype)

    indicator = indicator.to(torch.long)
    output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)

    x = x * output_image_mask
    x = transformer.input_proj(x) * output_image_mask

    t_cond = transformer.t_embedding(t)
    if t.dim() == 1:
        t_cond = t_cond.unsqueeze(1)
    adaln_input = F.silu(transformer.adaln_proj(t_cond))

    h = x

    image_indicator_embedding = transformer.embed_image_indicator(
        (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long)
    )
    h = h + image_indicator_embedding

    cos, sin = transformer.rotary_emb(position_ids)
    cos = cos.to(h.dtype)
    sin = sin.to(h.dtype)

    for layer in transformer.layers:
        h = layer(h, segment_ids=segment_ids, cos=cos, sin=sin, adaln_input=adaln_input)

    out = transformer.final_layer(h, c=adaln_input)
    return out.to(torch.float32)


class LowVramIdeogram4Pipeline:
    """Drop-in replacement for Ideogram4Pipeline that loads components on demand."""

    def __init__(self, weights_repo: str, quantization: str = "nf4", offline_mode: bool = False):
        from transformers import AutoTokenizer

        from ideogram4.latent_norm import get_latent_norm
        from ideogram4.pipeline_ideogram4 import Ideogram4PipelineConfig
        from modules.ideogram4.pipeline import _offline_env

        self.weights_repo = weights_repo
        self.quantization = quantization
        self.offline_mode = offline_mode
        self.config = Ideogram4PipelineConfig(weights_repo=weights_repo)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

        # Light, persistent state only (spec: keep tokenizer / config / latent norm).
        with _offline_env(offline_mode):
            self.text_tokenizer = AutoTokenizer.from_pretrained(weights_repo, subfolder=self.config.tokenizer_subfolder)
        shift, scale = get_latent_norm()
        self.latent_shift = shift.to(self.device)
        self.latent_scale = scale.to(self.device)

    # ---- component loaders (each freed by the caller after use) --------------
    def _load_text_encoder(self):
        from ideogram4.pipeline_ideogram4 import _load_qwen3_vl
        from modules.ideogram4.pipeline import _offline_env

        with _offline_env(self.offline_mode):
            _tok, encoder = _load_qwen3_vl(
                self.weights_repo, self.device, self.dtype,
                tokenizer_subfolder=self.config.tokenizer_subfolder,
                text_encoder_subfolder=self.config.text_encoder_subfolder,
            )
        return encoder

    def _load_transformers(self):
        from ideogram4.modeling_ideogram4 import Ideogram4Config
        from ideogram4.pipeline_ideogram4 import _build_transformer, _load_indexed_or_single_state_dict
        from modules.ideogram4.pipeline import _offline_env

        cfg = Ideogram4Config()
        with _offline_env(self.offline_mode):
            cond_sd = _load_indexed_or_single_state_dict(self.weights_repo, self.config.conditional_index_filename)
        conditional = _build_transformer(cfg, cond_sd, self.device, self.dtype)
        del cond_sd
        _free_cuda()

        with _offline_env(self.offline_mode):
            uncond_sd = _load_indexed_or_single_state_dict(self.weights_repo, self.config.unconditional_index_filename)
        unconditional = _build_transformer(cfg, uncond_sd, self.device, self.dtype)
        del uncond_sd
        _free_cuda()
        return conditional, unconditional

    def _load_autoencoder(self):
        from huggingface_hub import hf_hub_download

        from ideogram4.pipeline_ideogram4 import _load_autoencoder
        from modules.ideogram4.pipeline import _offline_env

        with _offline_env(self.offline_mode):
            weights_path = hf_hub_download(repo_id=self.weights_repo, filename=self.config.autoencoder_filename)
            return _load_autoencoder(weights_path, self.device, self.dtype)

    # ---- replicated helpers (verbatim-faithful to the official pipeline) -----
    def _tokenize(self, prompt: str):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.text_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        encoded = self.text_tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = encoded["input_ids"][0]
        num_text_tokens = int(token_ids.shape[0])
        if num_text_tokens > self.config.max_text_tokens:
            raise ValueError(f"prompt has {num_text_tokens} tokens, exceeds max_text_tokens={self.config.max_text_tokens}")
        return token_ids, num_text_tokens

    def _build_inputs(self, prompts, height, width):
        from ideogram4.constants import (
            IMAGE_POSITION_OFFSET,
            LLM_TOKEN_INDICATOR,
            OUTPUT_IMAGE_INDICATOR,
            SEQUENCE_PADDING_INDICATOR,
        )

        tokenized = [self._tokenize(p) for p in prompts]
        batch_size = len(prompts)

        patch = self.config.patch_size * self.config.ae_scale_factor
        if height % patch != 0 or width % patch != 0:
            raise ValueError(f"height/width must be divisible by patch_size*ae_scale_factor={patch}")
        grid_h = height // patch
        grid_w = width // patch
        num_image_tokens = grid_h * grid_w

        max_text_tokens = max(num_text for _, num_text in tokenized)
        total_seq_len = max_text_tokens + num_image_tokens

        h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
        t_idx = torch.zeros_like(h_idx)
        image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

        token_ids = torch.zeros(batch_size, total_seq_len, dtype=torch.long)
        text_position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
        position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
        segment_ids = torch.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
        indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

        for b, (toks, num_text) in enumerate(tokenized):
            pad_len = max_text_tokens - num_text
            total_unpadded = num_text + num_image_tokens
            offset = pad_len
            token_ids[b, offset : offset + num_text] = toks

            text_pos = torch.arange(num_text)
            text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
            text_position_ids[b, offset : offset + num_text] = text_pos_3d
            position_ids[b, offset : offset + num_text] = text_pos_3d
            position_ids[b, offset + num_text :] = image_pos

            indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
            indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR
            segment_ids[b, offset : offset + total_unpadded] = 1

        return {
            "token_ids": token_ids.to(self.device),
            "text_position_ids": text_position_ids.to(self.device),
            "position_ids": position_ids.to(self.device),
            "segment_ids": segment_ids.to(self.device),
            "indicator": indicator.to(self.device),
            "num_image_tokens": num_image_tokens,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "max_text_tokens": max_text_tokens,
        }

    def _encode_text_sparse(self, encoder, text_token_ids, text_pos_1d):
        """Text-only Qwen3-VL encode → stacked tap-layer features (B, T, llm_dim) float32.

        Mirrors the official _get_qwen3_vl_embeddings / _encode_text but on text tokens
        only (a causal decoder's text features don't depend on the image tokens).
        """
        from transformers.masking_utils import create_causal_mask

        from ideogram4.constants import QWEN3_VL_ACTIVATION_LAYERS

        language_model = encoder.language_model
        batch_size, seq_len = text_token_ids.shape

        inputs_embeds = language_model.embed_tokens(text_token_ids)
        attention_mask = torch.ones_like(text_token_ids, dtype=torch.long)

        pos_2d = text_pos_1d.contiguous()
        position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
        text_position_ids = position_ids_4d[0]
        mrope_position_ids = position_ids_4d[1:]

        causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

        tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
        captured = {}
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            if layer_idx in tap_set:
                captured[layer_idx] = hidden_states

        selected = [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]
        stacked = torch.stack(selected, dim=0)
        stacked = torch.permute(stacked, (1, 2, 3, 0))
        stacked = stacked.reshape(batch_size, seq_len, -1)
        return stacked.to(torch.float32)

    def _decode(self, autoencoder, z, *, grid_h, grid_w):
        batch_size = z.shape[0]
        patch = self.config.patch_size

        z = z * self.latent_scale + self.latent_shift

        ae_channels = z.shape[-1] // (patch * patch)
        z = z.view(batch_size, grid_h, grid_w, patch, patch, ae_channels)
        z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
        z = z.view(batch_size, ae_channels, grid_h * patch, grid_w * patch)

        z = z.to(self.dtype)
        decoded = autoencoder.decoder(z)

        from PIL import Image

        decoded = decoded.float().clamp(-1.0, 1.0)
        decoded = ((decoded + 1.0) * 127.5).round().to(torch.uint8)
        decoded = decoded.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(arr) for arr in decoded]

    # ---- generation ---------------------------------------------------------
    @torch.no_grad()
    def __call__(self, prompts, *, height=1024, width=1024, num_steps=128, guidance_scale=7.0,
                 guidance_schedule=None, mu=0.5, std=1.0, seed=None, step_callback=None, **_ignored):
        from ideogram4.modeling_ideogram4 import Ideogram4Config
        from ideogram4.scheduler import get_schedule_for_resolution, make_step_intervals

        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = prompts[:1]  # 16GB mode is batch size 1 (driver calls once per image)
        batch_size = 1

        schedule = get_schedule_for_resolution((height, width), known_mean=mu, std=std)
        step_intervals = make_step_intervals(num_steps).to(self.device)

        if guidance_schedule is not None:
            gw_per_step = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=self.device)
            if gw_per_step.shape != (num_steps,):
                raise ValueError(f"guidance_schedule must have shape ({num_steps},), got {tuple(gw_per_step.shape)}")
        else:
            gw_per_step = torch.full((num_steps,), float(guidance_scale), dtype=torch.float32, device=self.device)

        inputs = self._build_inputs(prompts, height=height, width=width)
        num_image_tokens = inputs["num_image_tokens"]
        grid_h, grid_w = inputs["grid_h"], inputs["grid_w"]
        max_text_tokens = inputs["max_text_tokens"]
        latent_dim = Ideogram4Config().in_channels

        # ---- phase 1: text encoder (load → text-only encode → free) ----------
        _vram("before text encoder")
        logger.info("Ideogram 4.0 low VRAM 16GB: load text encoder ...")
        encoder = self._load_text_encoder()
        _vram("text encoder loaded")
        text_token_ids = inputs["token_ids"][:, :max_text_tokens]
        text_pos_1d = inputs["text_position_ids"][:, :max_text_tokens, 0]
        llm_features_text = self._encode_text_sparse(encoder, text_token_ids, text_pos_1d)
        encoder = None
        _free_cuda()
        _vram("after text encode (encoder freed)")

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(int(seed))
        z = torch.randn(batch_size, num_image_tokens, latent_dim, dtype=torch.float32, device=self.device, generator=generator)
        text_z_padding = torch.zeros(batch_size, max_text_tokens, latent_dim, dtype=torch.float32, device=self.device)

        neg_position_ids = inputs["position_ids"][:, max_text_tokens:]
        neg_segment_ids = inputs["segment_ids"][:, max_text_tokens:]
        neg_indicator = inputs["indicator"][:, max_text_tokens:]

        # ---- phase 2: transformers (load both → denoise → free) --------------
        logger.info("Ideogram 4.0 low VRAM 16GB: load transformers ...")
        conditional, unconditional = self._load_transformers()
        _vram("transformers loaded")

        denoise_start = time.time()
        for i in range(num_steps - 1, -1, -1):
            if step_callback is not None:
                step_callback()

            t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
            s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
            t = torch.full((batch_size,), t_val, dtype=torch.float32, device=self.device)

            pos_z = torch.cat([text_z_padding, z], dim=1)
            pos_out = _conditional_forward_sparse(
                conditional,
                llm_features_text=llm_features_text,
                x=pos_z, t=t,
                position_ids=inputs["position_ids"],
                segment_ids=inputs["segment_ids"],
                indicator=inputs["indicator"],
                text_token_count=max_text_tokens,
            )
            pos_v = pos_out[:, max_text_tokens:]

            neg_v = _unconditional_forward_image_only(
                unconditional,
                x=z, t=t,
                position_ids=neg_position_ids,
                segment_ids=neg_segment_ids,
                indicator=neg_indicator,
            )

            gw_i = gw_per_step[i]
            v = gw_i * pos_v + (1.0 - gw_i) * neg_v
            z = z + v * (s_val - t_val)

        logger.info("Ideogram 4.0 low VRAM 16GB: denoise done in %.1fs", time.time() - denoise_start)
        conditional = None
        unconditional = None
        del llm_features_text, text_z_padding
        _free_cuda()
        _vram("after denoise (transformers freed)")

        # ---- phase 3: VAE (load → decode → free) -----------------------------
        logger.info("Ideogram 4.0 low VRAM 16GB: load autoencoder ...")
        autoencoder = self._load_autoencoder()
        _vram("autoencoder loaded")
        decode_start = time.time()
        images = self._decode(autoencoder, z, grid_h=grid_h, grid_w=grid_w)
        logger.info("Ideogram 4.0 low VRAM 16GB: decode done in %.1fs", time.time() - decode_start)
        autoencoder = None
        _free_cuda()
        _vram("after decode (autoencoder freed)")

        return images
