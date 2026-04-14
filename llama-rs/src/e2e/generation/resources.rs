//! Persistent GPU resource management for the decode phase.
//!
//! Contains builders and wrappers for GPU-resident resources:
//! - Persistent QKV projections (full + linear attention)
//! - Persistent KV caches
//! - Persistent MLP graphs
//! - LM head graph for greedy sampling
//! - [`PersistentDecodeResources`] — unified bundle of all decode resources

use super::LayerPassConfig;
use super::strategy::DecodeStrategy;
use crate::e2e::attention::{
    PersistentKvCache, PersistentScoringContext, QkvProjections, build_persistent_kv_cache,
    full_attention_decode_core, full_attention_hidden_features, prepare_qkv_from_raw,
};
use crate::e2e::error::{E2eError, GgmlResultExt};
use crate::e2e::linear_attention::{
    LinearDecodeScratch, LinearProjections, linear_attention_conv_channels,
    linear_attention_decode_core, linear_attention_hidden_features,
};
use crate::e2e::mlp::{PersistentMlp, build_persistent_mlp, mlp_sequence_inference_with_weights};
use crate::e2e::numeric::checked_mul;
use crate::e2e::plan::{
    AttentionLayerPlan, LayerPlan, Qwen35FullAttentionLayerPlan, Qwen35LinearAttentionLayerPlan,
};
use crate::e2e::state::{GenerationState, LayerAttentionState};
use crate::e2e::tensor_ops::{
    PersistentDecodeProjection, add_in_place, build_lm_head_graph,
    build_persistent_full_attention_graphs, build_persistent_linear_attention_graphs,
    lm_head_sample_step, recommended_lm_head_memory, recommended_persistent_full_attention_memory,
    recommended_persistent_linear_attention_memory, rms_norm_with_weight, upload_weight,
};
use ggml_rs::{Backend, Context};

type PersistentProjectionSets = (
    Vec<Option<Context>>,
    Vec<Option<PersistentDecodeProjection<'static>>>,
);

/// Attempt to build persistent projection contexts and graphs for all layers.
///
/// Returns `None` on any failure (opportunistic — caller falls back to
/// `DecodeStrategy`). On success returns `(contexts, projections)` where both
/// vecs are aligned to `layer_plans` and each slot is `Some` for attention
/// layers, `None` otherwise.
fn try_build_persistent_projections(
    layer_plans: &[LayerPlan],
    backend: &Backend,
) -> Option<PersistentProjectionSets> {
    // SAFETY reasoning for the 'static transmute below:
    //
    // `Tensor<'ctx, T>`, `Graph<'ctx>`, and `BackendBuffer<'ctx>` all carry
    // `PhantomData<&'ctx Context>` — they contain raw pointers, not actual
    // Rust references.  The lifetime exists solely so the borrow-checker can
    // enforce "Context outlives its derived handles" at the API boundary.
    //
    // Inside `two_phase_loop` we store `proj_ctxs` (the owning `Vec<Option<Context>>`)
    // and `decode_projs` (the derived handles) as sibling locals declared in
    // strict order so that `decode_projs` drops *before* `proj_ctxs`.
    // This maintains the real invariant: handles never outlive their context.
    //
    // The transmute erases the unnameable local borrow `'ctx` that would
    // otherwise require GATs or self-referential structs.  Because both
    // containers live for the same scope and drop in the correct order, this
    // is sound.

    let mut contexts: Vec<Option<Context>> = Vec::with_capacity(layer_plans.len());
    let mut projections: Vec<Option<PersistentDecodeProjection<'static>>> =
        Vec::with_capacity(layer_plans.len());

    for layer_plan in layer_plans {
        let attention = match &layer_plan.attention {
            Some(a) => a,
            None => {
                contexts.push(None);
                projections.push(None);
                continue;
            }
        };

        let result: Result<(PersistentDecodeProjection<'static>, Context), E2eError> =
            match attention {
                AttentionLayerPlan::Qwen35Full(attn) => build_one_persistent_full(attn, backend),
                AttentionLayerPlan::Qwen35Linear(attn) => {
                    build_one_persistent_linear(attn, backend)
                }
                AttentionLayerPlan::Standard(_) => {
                    // Standard attention doesn't yet have persistent projections;
                    // skip but don't block other layers.
                    contexts.push(None);
                    projections.push(None);
                    continue;
                }
            };

        match result {
            Ok((proj, ctx)) => {
                contexts.push(Some(ctx));
                projections.push(Some(proj));
            }
            Err(_) => return None, // Any failure → give up on persistent path
        }
    }

    debug_assert_eq!(contexts.len(), layer_plans.len());
    debug_assert_eq!(projections.len(), layer_plans.len());
    Some((contexts, projections))
}

fn build_one_persistent_full(
    attn: &Qwen35FullAttentionLayerPlan,
    backend: &Backend,
) -> Result<(PersistentDecodeProjection<'static>, Context), E2eError> {
    let hidden_features = full_attention_hidden_features(attn)?;
    let query_features = checked_mul(attn.head_count, attn.head_dimension)?;
    let kv_features = checked_mul(attn.kv_head_count, attn.head_dimension)?;
    let query_features_x2 = checked_mul(query_features, 2)?;

    let ctx_size = recommended_persistent_full_attention_memory(
        hidden_features,
        query_features_x2,
        kv_features,
        query_features,
    )?;
    let ctx = Context::new_no_alloc_bytes(ctx_size).ggml_ctx("Context(pfa)")?;

    let g = build_persistent_full_attention_graphs(
        &ctx,
        hidden_features,
        query_features_x2,
        kv_features,
        query_features,
    )?;

    let buffer = ctx.allocate_tensors(backend).ggml_ctx("allocate(pfa)")?;

    upload_weight(&g.w_q, &attn.q_weight_values, "write<W_Q>(pfa)")?;
    upload_weight(&g.w_k, &attn.k_weight_values, "write<W_K>(pfa)")?;
    upload_weight(&g.w_v, &attn.v_weight_values, "write<W_V>(pfa)")?;
    upload_weight(&g.output.w, &attn.output_weight_values, "write<W_OUT>(pfa)")?;

    // SAFETY: see the comment block in `try_build_persistent_projections`.
    let proj = unsafe {
        std::mem::transmute::<PersistentDecodeProjection<'_>, PersistentDecodeProjection<'static>>(
            PersistentDecodeProjection::FullAttention {
                x_in: g.x_in,
                q_out: g.q_out,
                k_out: g.k_out,
                v_out: g.v_out,
                input_graph: g.input_graph,
                output: g.output,
                _buffer: buffer,
            },
        )
    };
    Ok((proj, ctx))
}

fn build_one_persistent_linear(
    attn: &Qwen35LinearAttentionLayerPlan,
    backend: &Backend,
) -> Result<(PersistentDecodeProjection<'static>, Context), E2eError> {
    let hidden_features = linear_attention_hidden_features(attn)?;
    let conv_channels = linear_attention_conv_channels(attn)?;
    let inner_size = attn.inner_size;
    let time_step_rank = attn.time_step_rank;

    let ctx_size = recommended_persistent_linear_attention_memory(
        hidden_features,
        conv_channels,
        inner_size,
        time_step_rank,
    )?;
    let ctx = Context::new_no_alloc_bytes(ctx_size).ggml_ctx("Context(pla)")?;

    let g = build_persistent_linear_attention_graphs(
        &ctx,
        hidden_features,
        conv_channels,
        inner_size,
        time_step_rank,
    )?;

    let buffer = ctx.allocate_tensors(backend).ggml_ctx("allocate(pla)")?;

    upload_weight(&g.w_qkv, &attn.qkv_weight_values, "write<W_QKV>(pla)")?;
    upload_weight(&g.w_z, &attn.gate_weight_values, "write<W_Z>(pla)")?;
    upload_weight(&g.w_alpha, &attn.alpha_weight_values, "write<W_ALPHA>(pla)")?;
    upload_weight(&g.w_beta, &attn.beta_weight_values, "write<W_BETA>(pla)")?;
    upload_weight(
        &g.output.w,
        &attn.ssm_out_weight_values,
        "write<W_OUT>(pla)",
    )?;

    let proj = unsafe {
        std::mem::transmute::<PersistentDecodeProjection<'_>, PersistentDecodeProjection<'static>>(
            PersistentDecodeProjection::LinearAttention {
                x_in: g.x_in,
                qkv_out: g.qkv_out,
                z_out: g.z_out,
                alpha_out: g.alpha_out,
                beta_out: g.beta_out,
                input_graph: g.input_graph,
                output: g.output,
                _buffer: buffer,
            },
        )
    };
    Ok((proj, ctx))
}

type PersistentKvCacheSets = (
    Vec<Option<Context>>,
    Vec<Option<PersistentKvCache<'static>>>,
);

/// Attempt to build persistent backend-resident KV caches for all full
/// attention layers.
///
/// Returns `None` on any failure. Linear attention layers get `None` slots
/// (they have no quadratic KV cache). The `max_tokens` budget is shared
/// across all layers.
fn try_build_persistent_kv_caches(
    layer_plans: &[LayerPlan],
    max_tokens: usize,
    backend: &Backend,
) -> Option<PersistentKvCacheSets> {
    let mut contexts: Vec<Option<Context>> = Vec::with_capacity(layer_plans.len());
    let mut caches: Vec<Option<PersistentKvCache<'static>>> = Vec::with_capacity(layer_plans.len());

    for layer_plan in layer_plans {
        match &layer_plan.attention {
            Some(AttentionLayerPlan::Qwen35Full(attn)) => {
                match build_persistent_kv_cache(attn, max_tokens, backend) {
                    Ok((cache, ctx)) => {
                        contexts.push(Some(ctx));
                        caches.push(Some(cache));
                    }
                    Err(_) => return None,
                }
            }
            _ => {
                // Linear or missing — no persistent KV cache.
                contexts.push(None);
                caches.push(None);
            }
        }
    }

    debug_assert_eq!(contexts.len(), layer_plans.len());
    debug_assert_eq!(caches.len(), layer_plans.len());
    Some((contexts, caches))
}

type PersistentMlpSets = (Vec<Option<Context>>, Vec<Option<PersistentMlp<'static>>>);

/// Attempt to build persistent MLP graphs for all layers (per-layer opportunistic).
///
/// Each layer is built independently — if one fails, it gets `None` and falls
/// back to the ephemeral `mlp_sequence_inference_with_weights` path. Returns
/// aligned vecs (one slot per layer plan).
fn try_build_persistent_mlps(
    layer_plans: &[LayerPlan],
    rms_norm_eps: f32,
    backend: &Backend,
) -> PersistentMlpSets {
    // Transmute is handled inside `build_persistent_mlp` (same pattern as
    // `build_one_persistent_full`). Callers maintain LIFO drop order.

    let mut contexts: Vec<Option<Context>> = Vec::with_capacity(layer_plans.len());
    let mut mlps: Vec<Option<PersistentMlp<'static>>> = Vec::with_capacity(layer_plans.len());

    for layer_plan in layer_plans {
        let mlp_plan = &layer_plan.mlp;
        match build_persistent_mlp(
            &mlp_plan.weights,
            &mlp_plan.norm_values,
            rms_norm_eps,
            backend,
        ) {
            Ok((mlp, ctx)) => {
                contexts.push(Some(ctx));
                mlps.push(Some(mlp));
            }
            Err(_) => {
                contexts.push(None);
                mlps.push(None);
            }
        }
    }

    debug_assert_eq!(contexts.len(), layer_plans.len());
    debug_assert_eq!(mlps.len(), layer_plans.len());
    (contexts, mlps)
}

// ---------------------------------------------------------------------------
// LmHeadResources: persistent LM head graph — build once, sample per token
// ---------------------------------------------------------------------------

/// Persistent LM head graph resources for GPU-accelerated greedy sampling.
///
/// Built once from model weights; reused every decode step. Encapsulates the
/// ggml context, buffer, graph tensors, and graph in a self-contained struct
/// with correct drop ordering (tensors/graph before context/buffer).
pub(in crate::e2e) struct LmHeadResources {
    // Graph tensors referencing _ctx (dropped first)
    x_in: ggml_rs::Tensor<'static, f32>,
    logits_t: ggml_rs::Tensor<'static, f32>,
    graph: ggml_rs::Graph<'static>,
    // Context + buffer (dropped last — keep tensor memory alive)
    _buffer: ggml_rs::BackendBuffer<'static>,
    _ctx: Context,
}

impl LmHeadResources {
    /// Build persistent LM head graph and upload weights.
    ///
    /// Returns `None` if any step fails (context allocation, graph build,
    /// weight upload). Callers fall back to host-side greedy sampling.
    pub(in crate::e2e) fn try_build(
        hidden_features: usize,
        vocab_size: usize,
        rms_norm_eps: f32,
        output_weight_values: &[f32],
        output_norm_values: &[f32],
        backend: &Backend,
    ) -> Option<Self> {
        let ctx_size = recommended_lm_head_memory(hidden_features, vocab_size).ok()?;
        let ctx = Context::new_no_alloc_bytes(ctx_size).ok()?;
        let parts = build_lm_head_graph(&ctx, hidden_features, vocab_size, rms_norm_eps).ok()?;
        let buffer = ctx.allocate_tensors(backend).ok()?;
        parts.w_out.write_data_backend(output_weight_values).ok()?;
        parts.norm_w.write_data_backend(output_norm_values).ok()?;

        // SAFETY: ctx and buffer kept alive as struct fields; drop order
        // (declaration order, top→bottom) ensures tensors/graph drop before ctx/buffer.
        let x_in = unsafe {
            std::mem::transmute::<ggml_rs::Tensor<'_, f32>, ggml_rs::Tensor<'static, f32>>(
                parts.x_in,
            )
        };
        let logits_t = unsafe {
            std::mem::transmute::<ggml_rs::Tensor<'_, f32>, ggml_rs::Tensor<'static, f32>>(
                parts.logits,
            )
        };
        let graph = unsafe {
            std::mem::transmute::<ggml_rs::Graph<'_>, ggml_rs::Graph<'static>>(parts.graph)
        };
        let _buffer = unsafe {
            std::mem::transmute::<ggml_rs::BackendBuffer<'_>, ggml_rs::BackendBuffer<'static>>(
                buffer,
            )
        };

        Some(Self {
            x_in,
            logits_t,
            graph,
            _buffer,
            _ctx: ctx,
        })
    }

    /// Sample one token from a single hidden state vector via the GPU graph.
    pub(in crate::e2e) fn sample_hidden(
        &mut self,
        hidden_state: &[f32],
        backend: &Backend,
    ) -> Result<i32, E2eError> {
        lm_head_sample_step(
            hidden_state,
            &self.x_in,
            &self.logits_t,
            &mut self.graph,
            backend,
        )
    }
}

// ---------------------------------------------------------------------------
// PersistentDecodeResources: unified persistent resource bundle
// ---------------------------------------------------------------------------

/// All GPU-resident persistent resources needed for optimized decode.
///
/// Field ordering is **safety-critical**: resources that reference contexts must
/// be declared (and thus dropped) *before* the contexts that own the underlying
/// ggml memory.  Rust drops struct fields in declaration order (top to bottom).
///
/// The struct is built once after prefill and reused for every decode step.
/// Individual resource groups are independently optional — a failure in one
/// (e.g., KV cache) does not prevent others (e.g., MLPs) from being used.
pub(in crate::e2e) struct PersistentDecodeResources {
    // --- Resources referencing contexts (dropped first) ---
    pub scoring_ctx: Option<PersistentScoringContext>,
    pub linear_scratch: Option<LinearDecodeScratch>,
    pub persistent_mlps: Vec<Option<PersistentMlp<'static>>>,
    pub decode_projs: Option<Vec<Option<PersistentDecodeProjection<'static>>>>,
    pub kv_caches: Vec<Option<PersistentKvCache<'static>>>,

    // LM head (self-contained: owns its own context + buffer)
    lm_head: LmHeadResources,

    // --- Contexts (dropped last — keeps tensor memory alive) ---
    _mlp_ctxs: Vec<Option<Context>>,
    _proj_ctxs: Vec<Option<Context>>,
    _kv_ctxs: Vec<Option<Context>>,
}

impl PersistentDecodeResources {
    /// Build persistent decode resources from layer plans and model weights.
    ///
    /// The LM head must be pre-built by the caller. Projections, KV caches,
    /// scoring context, linear scratch, and MLPs are each independently
    /// optional — a failure in one category does not affect others.
    pub(in crate::e2e) fn try_build(
        layer_plans: &[LayerPlan],
        lm_head: LmHeadResources,
        rms_norm_eps: f32,
        total_sequence_length: usize,
        backend: &Backend,
    ) -> Self {
        // 1. Persistent MLPs (per-layer opportunistic)
        let (_mlp_ctxs, persistent_mlps) =
            try_build_persistent_mlps(layer_plans, rms_norm_eps, backend);

        // 3. Persistent projections (all-or-nothing per attention type)
        let persistent = try_build_persistent_projections(layer_plans, backend);
        let (_proj_ctxs, decode_projs) = match persistent {
            Some((ctxs, projs)) => (ctxs, Some(projs)),
            None => (Vec::new(), None),
        };

        // 4. Persistent KV caches (requires projections to be useful)
        let kv_persistent =
            try_build_persistent_kv_caches(layer_plans, total_sequence_length, backend);
        let (_kv_ctxs, kv_caches) = match kv_persistent {
            Some((ctxs, caches)) => (ctxs, caches),
            None => {
                let empty: Vec<Option<PersistentKvCache<'static>>> =
                    (0..layer_plans.len()).map(|_| None).collect();
                (Vec::new(), empty)
            }
        };

        // 5. Persistent scoring context (requires KV caches)
        let scoring_ctx = kv_caches
            .iter()
            .zip(layer_plans.iter())
            .find_map(|(kv, lp)| {
                if let (Some(kv), Some(AttentionLayerPlan::Qwen35Full(attn))) =
                    (kv, lp.attention.as_ref())
                {
                    Some((attn, kv))
                } else {
                    None
                }
            })
            .and_then(|(attn, kv)| {
                PersistentScoringContext::new(attn, total_sequence_length, kv, backend).ok()
            });

        // 6. Linear attention scratch buffers
        let linear_scratch = layer_plans.iter().find_map(|lp| {
            if let Some(AttentionLayerPlan::Qwen35Linear(attn)) = lp.attention.as_ref() {
                Some(LinearDecodeScratch::new(attn.state_size, attn.inner_size))
            } else {
                None
            }
        });

        Self {
            scoring_ctx,
            linear_scratch,
            persistent_mlps,
            decode_projs,
            kv_caches,
            lm_head,
            _mlp_ctxs,
            _proj_ctxs,
            _kv_ctxs,
        }
    }

    /// Seed persistent KV caches from host-side prefill state.
    pub(in crate::e2e) fn seed_kv_caches(&self, state: &GenerationState) {
        for (layer_idx, cache) in self.kv_caches.iter().enumerate() {
            if let (Some(cache), Some(LayerAttentionState::Qwen35Full(s))) =
                (cache, state.layers.get(layer_idx))
            {
                let _ = cache.seed_from_host(&s.k_cache, &s.v_cache, s.token_count());
            }
        }
    }

    /// Run one LM head sampling step on a single-token hidden state.
    pub(in crate::e2e) fn sample_token(
        &mut self,
        hidden: &[f32],
        token_index: usize,
        hidden_features: usize,
        backend: &Backend,
    ) -> Result<i32, E2eError> {
        let offset = checked_mul(token_index, hidden_features)?;
        let last_hidden = &hidden[offset..offset + hidden_features];
        self.lm_head.sample_hidden(last_hidden, backend)
    }

    /// Run one decode step through all layers using persistent resources.
    ///
    /// If persistent projections are available, uses the fast path
    /// (inline persistent decode). Otherwise falls back to
    /// `DecodeStrategy` with `process_all_layers`.
    pub(in crate::e2e) fn decode_step(
        &mut self,
        hidden: &mut [f32],
        config: &LayerPassConfig<'_>,
        state: &mut GenerationState,
        hidden_features: usize,
    ) -> Result<(), E2eError> {
        if self.decode_projs.is_some() {
            self.persistent_decode_all_layers(
                hidden,
                config.layer_plans,
                state,
                hidden_features,
                config.rms_norm_eps,
                config.backend,
            )
        } else {
            // Fallback: per-token weight upload for attention, but persistent MLPs
            let mut strategy = DecodeStrategy { state };
            super::process_all_layers(hidden, config, &mut strategy, 1, &mut self.persistent_mlps)
        }
    }

    /// Process all layers in decode mode using persistent projections.
    ///
    /// For each layer: host norm → persistent input proj → core logic → persistent
    /// output proj → residual add → persistent or ephemeral MLP.
    ///
    /// Caller must ensure `self.decode_projs.is_some()`.
    fn persistent_decode_all_layers(
        &mut self,
        hidden: &mut [f32],
        layer_plans: &[LayerPlan],
        state: &mut GenerationState,
        hidden_features: usize,
        rms_norm_eps: f32,
        backend: &Backend,
    ) -> Result<(), E2eError> {
        let projections = self
            .decode_projs
            .as_mut()
            .ok_or(E2eError::UnsupportedTwoPhase)?;

        debug_assert_eq!(layer_plans.len(), projections.len());
        debug_assert_eq!(layer_plans.len(), self.kv_caches.len());
        debug_assert!(
            self.persistent_mlps.is_empty() || self.persistent_mlps.len() == layer_plans.len(),
            "persistent_mlps must be empty (disabled) or aligned to layer_plans"
        );

        for (layer_idx, layer_plan) in layer_plans.iter().enumerate() {
            if let (Some(attention), Some(proj)) =
                (&layer_plan.attention, projections[layer_idx].as_mut())
            {
                let norm_weight = attention.norm_values();
                let normalized =
                    rms_norm_with_weight(hidden, hidden_features, 1, norm_weight, rms_norm_eps)?;

                proj.project_input(&normalized, backend)?;

                let attention_output = match (attention, &mut state.layers[layer_idx]) {
                    (AttentionLayerPlan::Qwen35Full(attn), LayerAttentionState::Qwen35Full(s)) => {
                        let QkvProjections {
                            q_full,
                            k_proj,
                            v_proj,
                        } = proj.read_full_attention_projections()?;
                        let hf = full_attention_hidden_features(attn)?;
                        let prepared = prepare_qkv_from_raw(
                            attn,
                            q_full,
                            k_proj,
                            v_proj,
                            1,
                            hf,
                            rms_norm_eps,
                        )?;
                        let sc = self.scoring_ctx.as_mut();
                        let head_outputs = full_attention_decode_core(
                            prepared,
                            attn,
                            s,
                            Some(backend),
                            self.kv_caches[layer_idx].as_ref(),
                            sc,
                        )?;
                        proj.project_output(&head_outputs, backend)?
                    }
                    (
                        AttentionLayerPlan::Qwen35Linear(attn),
                        LayerAttentionState::Qwen35Linear(s),
                    ) => {
                        let raw = proj.read_linear_attention_projections()?;
                        let conv_channels = linear_attention_conv_channels(attn)?;
                        let hf = linear_attention_hidden_features(attn)?;
                        let projections = LinearProjections {
                            qkv: raw.qkv,
                            z: raw.z,
                            alpha: raw.alpha,
                            beta: raw.beta,
                            conv_channels,
                            hidden_features: hf,
                        };
                        let output = linear_attention_decode_core(
                            projections,
                            attn,
                            rms_norm_eps,
                            s,
                            self.linear_scratch.as_mut(),
                        )?;
                        proj.project_output(&output, backend)?
                    }
                    _ => return Err(E2eError::UnsupportedTwoPhase),
                };
                add_in_place(hidden, &attention_output)?;
            }

            let mlp_output = if let Some(Some(mlp)) = self.persistent_mlps.get_mut(layer_idx) {
                mlp.step(hidden, backend)?
            } else {
                mlp_sequence_inference_with_weights(
                    &layer_plan.mlp.weights,
                    hidden,
                    1,
                    &layer_plan.mlp.norm_values,
                    rms_norm_eps,
                    backend,
                )?
            };
            add_in_place(hidden, &mlp_output)?;
        }
        Ok(())
    }
}
