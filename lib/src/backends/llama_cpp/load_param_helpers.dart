// Pure helpers for the load path. Kept here (vs inlined in the service) so
// they can be unit-tested without going through `LlamaEngine.loadModel`,
// which is integration-level and needs a real model file.

import '../../core/models/config/flash_attention.dart';
import '../../core/models/config/kv_cache_type.dart';
import '../../core/models/inference/model_params.dart';
import 'bindings.dart';

/// Maps llamadart's [KvCacheType] enum to llama.cpp's `ggml_type`. Pure
/// switch, no side effects.
ggml_type ggmlTypeFor(KvCacheType type) {
  switch (type) {
    case KvCacheType.f16:
      return ggml_type.GGML_TYPE_F16;
    case KvCacheType.q8_0:
      return ggml_type.GGML_TYPE_Q8_0;
    case KvCacheType.q4_0:
      return ggml_type.GGML_TYPE_Q4_0;
  }
}

/// Resolves the user-requested [FlashAttention] given the requested KV
/// cache types. llama.cpp refuses non-F16 KV without flash attention, so
/// `auto` is auto-promoted to `enabled` when either KV type isn't F16.
/// Explicit `enabled` / `disabled` are passed through unchanged.
///
/// Pairing this with [ModelParams]'s constructor-side ArgumentError on
/// `(non-F16 KV, FA disabled)` ensures the only ambiguous case (`auto`)
/// gets resolved deterministically here.
FlashAttention resolveFlashAttention({
  required FlashAttention requested,
  required KvCacheType cacheTypeK,
  required KvCacheType cacheTypeV,
}) {
  final wantsKvQuantization =
      cacheTypeK != KvCacheType.f16 || cacheTypeV != KvCacheType.f16;
  if (requested == FlashAttention.auto && wantsKvQuantization) {
    return FlashAttention.enabled;
  }
  return requested;
}

/// Applies the user-controlled fields of [params] to a freshly-defaulted
/// `llama_model_params` struct. Pure function: caller is responsible for
/// initialising and freeing the struct.
void applyModelParams(llama_model_params mparams, ModelParams params) {
  mparams.use_mmap = params.useMmap;
  mparams.use_mlock = params.useMlock;
}

/// Applies the user-controlled fields of [params] to a `llama_context_params`
/// struct. Honours the `auto` → `enabled` flash-attention promotion via
/// [resolveFlashAttention]. Returns the resolved [FlashAttention] so the
/// caller can log whether a promotion occurred.
FlashAttention applyContextParams(
  llama_context_params ctxParams,
  ModelParams params,
) {
  final resolvedFlashAttn = resolveFlashAttention(
    requested: params.flashAttention,
    cacheTypeK: params.cacheTypeK,
    cacheTypeV: params.cacheTypeV,
  );
  switch (resolvedFlashAttn) {
    case FlashAttention.auto:
      break;
    case FlashAttention.enabled:
      ctxParams.flash_attn_typeAsInt =
          llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_ENABLED.value;
      break;
    case FlashAttention.disabled:
      ctxParams.flash_attn_typeAsInt =
          llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_DISABLED.value;
      break;
  }
  ctxParams.type_kAsInt = ggmlTypeFor(params.cacheTypeK).value;
  ctxParams.type_vAsInt = ggmlTypeFor(params.cacheTypeV).value;
  if (params.kvUnified != null) {
    ctxParams.kv_unified = params.kvUnified!;
  }
  if (params.ropeFrequencyBase != null) {
    ctxParams.rope_freq_base = params.ropeFrequencyBase!;
  }
  if (params.ropeFrequencyScale != null) {
    ctxParams.rope_freq_scale = params.ropeFrequencyScale!;
  }
  return resolvedFlashAttn;
}
