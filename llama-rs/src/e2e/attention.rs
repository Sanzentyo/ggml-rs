//! Qwen3.5 full (standard-style) attention with gated Q and NeoX RoPE.
//!
//! Provides full-sequence inference, prefill (capturing KV cache), and
//! single-token decode step using cached state.

use super::error::E2eError;
use super::numeric::{checked_mul, dot, sigmoid_scalar, softmax_prefix};
use super::plan::Qwen35FullAttentionLayerPlan;
use super::state::Qwen35FullAttentionState;
use super::tensor_ops::{
    PROJECTION_SLACK_BYTES, per_head_rms_norm, project_sequence, project_sequence_graph,
};
use ggml_rs::{Backend, BackendBuffer, Bytes, Context, GraphAllocator, Shape2D, Shape4D, Tensor};

/// Backend-resident KV cache for GPU-accelerated attention scoring.
///
/// Pre-allocates max-size K/V tensors on the backend. Each decode step:
/// - Appends only the new token's K/V via `write_data_backend_at` (O(1))
/// - Ephemeral scoring graph creates `view_4d_of` into these tensors
///
/// Host `Vec<f32>` KV cache in `Qwen35FullAttentionState` remains the
/// source of truth (for checkpoint serialization and host-scoring fallback).
/// This struct is runtime-only decode infrastructure — not serializable.
pub(super) struct PersistentKvCache<'ctx> {
    /// K values: `[D, MaxT, Hkv, 1]` on backend (flash-friendly time-major layout).
    k_tensor: Tensor<'ctx, f32>,
    /// V values: `[D, MaxT, Hkv, 1]` on backend (flash-friendly time-major layout).
    v_tensor: Tensor<'ctx, f32>,
    /// Backend buffer keeping tensors alive on device.
    _buffer: BackendBuffer<'ctx>,
    /// Head dimension (D).
    head_dim: usize,
    /// Number of KV heads (Hkv).
    kv_head_count: usize,
    /// Max tokens this cache can hold.
    max_tokens: usize,
}

impl<'ctx> PersistentKvCache<'ctx> {
    /// Append one token's K/V to the backend tensors at the given position.
    ///
    /// `cached_len` is the position index (0-based) of the new token. The
    /// caller must ensure `cached_len < max_tokens`.
    ///
    /// Layout is `[D, MaxT, Hkv, 1]` (flash-friendly time-major). Each KV
    /// head's D-element slice is written to the appropriate stride offset.
    pub(super) fn append_token(
        &self,
        k_values: &[f32],
        v_values: &[f32],
        cached_len: usize,
    ) -> Result<(), E2eError> {
        if cached_len >= self.max_tokens {
            return Err(E2eError::SequenceTooLong {
                requested: cached_len + 1,
                context_length: self.max_tokens,
            });
        }
        let d = self.head_dim;
        let max_t = self.max_tokens;
        // In [D, MaxT, Hkv, 1] layout:
        // - stride between time positions = D elements
        // - stride between heads = D * MaxT elements
        // - offset for head h, position t = h * D * MaxT + t * D
        for h in 0..self.kv_head_count {
            let src_offset = h * d;
            let dst_offset = h * d * max_t + cached_len * d;
            self.k_tensor
                .write_data_backend_at(dst_offset, &k_values[src_offset..src_offset + d])
                .map_err(|source| E2eError::ggml("append_token(K)", source))?;
            self.v_tensor
                .write_data_backend_at(dst_offset, &v_values[src_offset..src_offset + d])
                .map_err(|source| E2eError::ggml("append_token(V)", source))?;
        }
        Ok(())
    }

    /// Bulk-upload existing KV prefix from host cache (used at initialization
    /// after prefill populates the host cache).
    ///
    /// Host cache layout is `[D * Hkv, T]` (contiguous per-token), which must
    /// be transposed to `[D, MaxT, Hkv, 1]` (per-head, time-major).
    pub(super) fn seed_from_host(
        &self,
        k_cache: &[f32],
        v_cache: &[f32],
        cached_len: usize,
    ) -> Result<(), E2eError> {
        let d = self.head_dim;
        let hkv = self.kv_head_count;
        let max_t = self.max_tokens;
        let kv_features = d * hkv;

        for t in 0..cached_len {
            let src_base = t * kv_features;
            for h in 0..hkv {
                let src_offset = src_base + h * d;
                let dst_offset = h * d * max_t + t * d;
                self.k_tensor
                    .write_data_backend_at(dst_offset, &k_cache[src_offset..src_offset + d])
                    .map_err(|source| E2eError::ggml("seed(K)", source))?;
                self.v_tensor
                    .write_data_backend_at(dst_offset, &v_cache[src_offset..src_offset + d])
                    .map_err(|source| E2eError::ggml("seed(V)", source))?;
            }
        }
        Ok(())
    }
}

/// Build a persistent backend-resident KV cache for one full attention layer.
///
/// Returns `(Context, PersistentKvCache<'static>)`. The context must be stored
/// in a parallel container and outlive the cache handles (same pattern as
/// `PersistentDecodeProjection`).
pub(super) fn build_persistent_kv_cache(
    attention: &Qwen35FullAttentionLayerPlan,
    max_tokens: usize,
    backend: &Backend,
) -> Result<(Context, PersistentKvCache<'static>), E2eError> {
    let d = attention.head_dimension;
    let hkv = attention.kv_head_count;

    // Two 4D tensors: K + V, each [D, MaxT, Hkv, 1] (flash-friendly time-major).
    // Metadata context: ~256 KB is plenty for 2 tensors.
    let ctx = Context::new_no_alloc(256 * 1024)
        .map_err(|source| E2eError::ggml("Context(persistent_kv)", source))?;

    let k_tensor = ctx
        .new_tensor_4d::<f32>(Shape4D::new(d, max_tokens, hkv, 1))
        .map_err(|source| E2eError::ggml("k_tensor(persistent_kv)", source))?;
    let v_tensor = ctx
        .new_tensor_4d::<f32>(Shape4D::new(d, max_tokens, hkv, 1))
        .map_err(|source| E2eError::ggml("v_tensor(persistent_kv)", source))?;

    let buffer = ctx
        .allocate_tensors(backend)
        .map_err(|source| E2eError::ggml("allocate(persistent_kv)", source))?;

    // SAFETY: same argument as `PersistentDecodeProjection` — the Context is
    // stored in a sibling container that drops after this struct. The lifetime
    // parameter on `Tensor`/`BackendBuffer` is PhantomData-only; the real
    // invariant (context outlives handles) is maintained by declaration order
    // in the caller.
    let cache = unsafe {
        std::mem::transmute::<PersistentKvCache<'_>, PersistentKvCache<'static>>(
            PersistentKvCache {
                k_tensor,
                v_tensor,
                _buffer: buffer,
                head_dim: d,
                kv_head_count: hkv,
                max_tokens,
            },
        )
    };

    Ok((ctx, cache))
}

/// Pre-reserved graph allocator for GPU-accelerated attention scoring.
///
/// Eliminates per-step Metal buffer allocation by pre-reserving a buffer
/// large enough for the maximum-T scoring graph. Each step reuses this
/// buffer via [`GraphAllocator::alloc_graph`] instead of calling
/// [`Context::allocate_tensors`].
///
/// One `PersistentScoringContext` can be shared across all full-attention
/// layers with the same dimensions (decode runs layers serially, so the
/// buffer is safely reused).
pub(super) struct PersistentScoringContext {
    gallocr: GraphAllocator,
}

impl PersistentScoringContext {
    /// Build a persistent scoring context by reserving a buffer for the
    /// worst-case (max_tokens) scoring graph topology.
    pub(super) fn new(
        attention: &Qwen35FullAttentionLayerPlan,
        max_tokens: usize,
        kv_cache: &PersistentKvCache<'static>,
        backend: &Backend,
    ) -> Result<Self, E2eError> {
        let mut gallocr = GraphAllocator::new(backend)
            .map_err(|source| E2eError::ggml("GraphAllocator::new(scoring)", source))?;

        // Build a reservation graph at max_tokens to determine buffer size.
        let ctx = Context::new_no_alloc(2 * 1024 * 1024)
            .map_err(|source| E2eError::ggml("ctx(scoring_reserve)", source))?;
        let sg = build_scoring_graph(&ctx, attention, max_tokens, kv_cache)?;

        gallocr
            .reserve(&sg.graph)
            .map_err(|source| E2eError::ggml("gallocr.reserve(scoring)", source))?;

        Ok(Self { gallocr })
    }
}

/// Intermediate result from [`build_scoring_graph`] holding all tensors
/// needed for data upload and result readback.
struct ScoringGraph<'ctx> {
    q: Tensor<'ctx, f32>,
    gate: Tensor<'ctx, f32>,
    gated: Tensor<'ctx, f32>,
    graph: ggml_rs::Graph<'ctx>,
}

/// Build the scoring compute graph for a given number of tokens.
///
/// The graph's tensors are not yet allocated — the caller must use either
/// `allocate_tensors` or `GraphAllocator::alloc_graph` before writing data
/// and computing.
fn build_scoring_graph<'ctx>(
    ctx: &'ctx Context,
    attention: &Qwen35FullAttentionLayerPlan,
    total_tokens: usize,
    kv_cache: &PersistentKvCache<'static>,
) -> Result<ScoringGraph<'ctx>, E2eError> {
    let d = attention.head_dimension;
    let h = attention.head_count;
    let hkv = attention.kv_head_count;
    let t = total_tokens;
    let elem = std::mem::size_of::<f32>();

    let q = ctx
        .new_tensor_4d::<f32>(Shape4D::new(d, 1, h, 1))
        .map_err(|source| E2eError::ggml("q_tensor", source))?;
    let gate = ctx
        .new_tensor_4d::<f32>(Shape4D::new(d, h, 1, 1))
        .map_err(|source| E2eError::ggml("gate_tensor", source))?;

    // Direct view into flash-friendly [D, MaxT, Hkv, 1] layout.
    // View [D, T, Hkv, 1] — already in the format flash_attn_ext expects,
    // no permute or contiguous copy needed.
    let max_t = kv_cache.max_tokens;
    let nb1 = d * elem; // byte stride between time positions
    let nb2 = d * max_t * elem; // byte stride between KV heads
    let nb3 = nb2 * hkv; // byte stride for dim3 (trivial)
    let k = ctx
        .view_4d_of(&kv_cache.k_tensor, d, t, hkv, 1, nb1, nb2, nb3, 0)
        .map_err(|source| E2eError::ggml("view_4d_of(K)", source))?;
    let v = ctx
        .view_4d_of(&kv_cache.v_tensor, d, t, hkv, 1, nb1, nb2, nb3, 0)
        .map_err(|source| E2eError::ggml("view_4d_of(V)", source))?;

    let attn = ctx
        .flash_attn_ext(&q, &k, &v, None, attention.attention_scale, 0.0, 0.0)
        .map_err(|source| E2eError::ggml("flash_attn_ext", source))?;

    let gate_sig = ctx
        .sigmoid(&gate)
        .map_err(|source| E2eError::ggml("sigmoid(gate)", source))?;
    let gated = ctx
        .mul(&attn, &gate_sig)
        .map_err(|source| E2eError::ggml("mul(attn,gate)", source))?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("new_graph(scoring)", source))?;
    graph.build_forward_expand(&gated);

    Ok(ScoringGraph {
        q,
        gate,
        gated,
        graph,
    })
}

/// GPU-accelerated scoring using persistent backend-resident KV cache.
///
/// Instead of uploading the entire KV cache each step, creates cross-context
/// views into the persistent tensors for the active prefix `[D, Hkv, T]`.
///
/// When a `PersistentScoringContext` is provided, its pre-reserved buffer is
/// reused (no per-step Metal buffer allocation). Otherwise, falls back to
/// `allocate_tensors` per step.
fn decode_scoring_gpu_persistent(
    q_values: &[f32],
    q_gate: &[f32],
    total_tokens: usize,
    attention: &Qwen35FullAttentionLayerPlan,
    kv_cache: &PersistentKvCache<'static>,
    backend: &Backend,
    scoring_ctx: Option<&mut PersistentScoringContext>,
) -> Result<Vec<f32>, E2eError> {
    let query_features = attention.head_dimension * attention.head_count;

    // Ephemeral scoring context — creates views into persistent KV tensors.
    let ctx = Context::new_no_alloc(2 * 1024 * 1024)
        .map_err(|source| E2eError::ggml("ctx(scoring_persistent)", source))?;

    let mut sg = build_scoring_graph(&ctx, attention, total_tokens, kv_cache)?;

    // Allocate: use pre-reserved gallocr if available, otherwise per-step alloc.
    let _buffer = if let Some(sc) = scoring_ctx {
        sc.gallocr
            .alloc_graph(&mut sg.graph)
            .map_err(|source| E2eError::ggml("gallocr.alloc_graph(scoring)", source))?;
        None
    } else {
        Some(
            ctx.allocate_tensors(backend)
                .map_err(|source| E2eError::ggml("allocate(scoring_persistent)", source))?,
        )
    };

    // Upload only Q and gate — K/V are already on device.
    sg.q.write_data_backend(q_values)
        .map_err(|source| E2eError::ggml("write(Q)", source))?;
    sg.gate
        .write_data_backend(q_gate)
        .map_err(|source| E2eError::ggml("write(gate)", source))?;

    backend
        .compute(&mut sg.graph)
        .map_err(|source| E2eError::ggml("compute(scoring_persistent)", source))?;

    let outputs = sg
        .gated
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read(gated)", source))?;

    if outputs.len() != query_features {
        return Err(E2eError::BufferLengthMismatch {
            expected: query_features,
            actual: outputs.len(),
        });
    }

    Ok(outputs)
}

pub(super) fn qwen35_full_attention_inference(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    attn_norm_weight: &[f32],
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    qwen35_full_attention_core(
        attention,
        input,
        sequence_length,
        rms_norm_eps,
        attn_norm_weight,
        None,
        backend,
    )
}

/// Prefill variant: computes full attention AND stores post-RoPE K + raw V in `state`.
pub(super) fn qwen35_full_attention_prefill(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    attn_norm_weight: &[f32],
    state: &mut Qwen35FullAttentionState,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    qwen35_full_attention_core(
        attention,
        input,
        sequence_length,
        rms_norm_eps,
        attn_norm_weight,
        Some(state),
        backend,
    )
}

/// Projected and normalized Q, K, V + gate vectors (pre-RoPE).
pub(super) struct PreparedAttention {
    pub(super) q_values: Vec<f32>,
    pub(super) k_values: Vec<f32>,
    pub(super) v_proj: Vec<f32>,
    pub(super) q_gate: Vec<f32>,
    pub(super) hidden_features: usize,
    pub(super) query_features: usize,
}

/// Estimate the backend memory needed for attention QKV projections.
///
/// Sums the memory for three independent matmuls (Q, K, V) sharing the same
/// input tensor, plus a slack constant for graph/tensor overhead.
fn recommended_qkv_projection_memory(
    hidden_features: usize,
    query_features_x2: usize,
    kv_features: usize,
    sequence_length: usize,
) -> Result<Bytes, E2eError> {
    let q_mem = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, query_features_x2),
        Shape2D::new(hidden_features, sequence_length),
    )
    .map_err(|source| E2eError::ggml("recommended_backend_matmul_memory(Q)", source))?;
    let k_mem = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, kv_features),
        Shape2D::new(hidden_features, sequence_length),
    )
    .map_err(|source| E2eError::ggml("recommended_backend_matmul_memory(K)", source))?;
    let v_mem = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, kv_features),
        Shape2D::new(hidden_features, sequence_length),
    )
    .map_err(|source| E2eError::ggml("recommended_backend_matmul_memory(V)", source))?;
    let total = q_mem
        .get()
        .checked_add(k_mem.get())
        .and_then(|v| v.checked_add(v_mem.get()))
        .and_then(|v| v.checked_add(PROJECTION_SLACK_BYTES))
        .ok_or(E2eError::MemorySizeOverflow)?;
    Ok(Bytes::new(total))
}

/// Host-side QKV projection outputs for full attention.
///
/// Used by both the fused graph builder (`project_qkv_graph`) and the
/// persistent projection reader (`read_full_attention_projections`).
#[derive(Debug)]
pub(super) struct QkvProjections {
    pub(super) q_full: Vec<f32>,
    pub(super) k_proj: Vec<f32>,
    pub(super) v_proj: Vec<f32>,
}

/// Validated derived dimensions for full (gated) attention.
///
/// Constructed once from `Qwen35FullAttentionLayerPlan`, validates all
/// dimension invariants upfront: GQA divisibility, hidden size consistency,
/// and weight buffer lengths.
#[derive(Debug, Clone, Copy)]
struct FullAttentionDims {
    /// Per-head feature dimension (D).
    d: usize,
    /// Number of query/gate heads (H).
    h: usize,
    /// Number of KV heads (Hkv ≤ H, for GQA).
    hkv: usize,
    /// Model hidden size (`H * D`, derived from output weight matrix).
    hidden: usize,
    /// Total query features (`H * D`).
    qf: usize,
    /// Total Q+gate interleaved features (`H * D * 2`).
    qf2: usize,
    /// Total KV features per tensor (`Hkv * D`).
    kvf: usize,
}

impl FullAttentionDims {
    fn new(attention: &Qwen35FullAttentionLayerPlan) -> Result<Self, E2eError> {
        let d = attention.head_dimension;
        let h = attention.head_count;
        let hkv = attention.kv_head_count;

        let hidden = h
            .checked_mul(d)
            .and_then(|qf| attention.output_weight_values.len().checked_div(qf))
            .filter(|&hid| hid > 0 && hid * h * d == attention.output_weight_values.len())
            .ok_or(E2eError::BufferLengthMismatch {
                expected: 1,
                actual: 0,
            })?;

        if !h.is_multiple_of(hkv) {
            return Err(E2eError::BufferLengthMismatch {
                expected: 0,
                actual: h % hkv,
            });
        }

        let qf = h * d;
        let qf2 = qf * 2;
        let kvf = hkv * d;

        Ok(Self {
            d,
            h,
            hkv,
            hidden,
            qf,
            qf2,
            kvf,
        })
    }

    /// Conservative memory estimate for the fully-fused attention graph.
    fn estimate_memory(&self, t: usize) -> Bytes {
        let Self {
            d,
            h: _,
            hkv: _,
            hidden,
            qf,
            qf2,
            kvf,
        } = *self;
        let weight_bytes = (hidden * qf2 + hidden * kvf * 2 + qf * hidden + d * 2 + hidden) * 4;
        let data_bytes = (hidden * t * 2
            + qf2 * t
            + kvf * t * 2
            + qf * t * 4
            + kvf * t * 2
            + qf * t * 3
            + t * t
            + hidden * t)
            * 4;
        let mask_bytes_estimate = t * t * 2; // f16
        Bytes::new((weight_bytes + data_bytes + mask_bytes_estimate) * 2 + 1_048_576)
    }
}

/// Compute Q, K, V projections using a single ggml compute graph.
///
/// Batches three `mul_mat` operations sharing the same input tensor into one
/// graph execution. Returns `(q_full, k_proj, v_proj)` as host vectors.
fn project_qkv_graph(
    input: &[f32],
    sequence_length: usize,
    hidden_features: usize,
    query_features_x2: usize,
    kv_features: usize,
    attention: &Qwen35FullAttentionLayerPlan,
    backend: &Backend,
) -> Result<QkvProjections, E2eError> {
    let ctx_size = recommended_qkv_projection_memory(
        hidden_features,
        query_features_x2,
        kv_features,
        sequence_length,
    )?;
    let ctx = Context::new_no_alloc_bytes(ctx_size)
        .map_err(|source| E2eError::ggml("Context::new_no_alloc_bytes(QKV)", source))?;

    let w_q = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, query_features_x2))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_Q>", source))?;
    let w_k = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, kv_features))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_K>", source))?;
    let w_v = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, kv_features))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_V>", source))?;
    let x = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, sequence_length))
        .map_err(|source| E2eError::ggml("new_tensor_2d<X>", source))?;

    let q_out = ctx
        .mul_mat(&w_q, &x)
        .map_err(|source| E2eError::ggml("mul_mat(Q)", source))?;
    let k_out = ctx
        .mul_mat(&w_k, &x)
        .map_err(|source| E2eError::ggml("mul_mat(K)", source))?;
    let v_out = ctx
        .mul_mat(&w_v, &x)
        .map_err(|source| E2eError::ggml("mul_mat(V)", source))?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("new_graph(QKV)", source))?;
    graph.build_forward_expand(&q_out);
    graph.build_forward_expand(&k_out);
    graph.build_forward_expand(&v_out);

    let _buffer = ctx
        .allocate_tensors(backend)
        .map_err(|source| E2eError::ggml("allocate_tensors(QKV)", source))?;

    w_q.write_data_backend(&attention.q_weight_values)
        .map_err(|source| E2eError::ggml("write_data_backend<W_Q>", source))?;
    w_k.write_data_backend(&attention.k_weight_values)
        .map_err(|source| E2eError::ggml("write_data_backend<W_K>", source))?;
    w_v.write_data_backend(&attention.v_weight_values)
        .map_err(|source| E2eError::ggml("write_data_backend<W_V>", source))?;
    x.write_data_backend(input)
        .map_err(|source| E2eError::ggml("write_data_backend<X>", source))?;

    backend
        .compute(&mut graph)
        .map_err(|source| E2eError::ggml("compute(QKV)", source))?;

    let q_full = q_out
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read_data_backend<Q>", source))?;
    let k_proj = k_out
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read_data_backend<K>", source))?;
    let v_proj = v_out
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read_data_backend<V>", source))?;
    Ok(QkvProjections {
        q_full,
        k_proj,
        v_proj,
    })
}

/// Shared projection + deinterleave + per-head RMS norm for both core and
/// decode paths. The caller applies RoPE with its own position_offset.
///
/// When `backend` is `Some`, uses ggml compute graphs for projections
/// (prefill/inference path). When `None`, falls back to host-side scalar
/// dot products (decode path, where graph overhead exceeds benefit).
/// Derive `hidden_features` from the output weight matrix dimensions.
pub(super) fn full_attention_hidden_features(
    attention: &Qwen35FullAttentionLayerPlan,
) -> Result<usize, E2eError> {
    let query_features = attention
        .head_count
        .checked_mul(attention.head_dimension)
        .ok_or(E2eError::MemorySizeOverflow)?;
    attention
        .output_weight_values
        .len()
        .checked_div(query_features)
        .filter(|&h| h > 0 && h * query_features == attention.output_weight_values.len())
        .ok_or(E2eError::BufferLengthMismatch {
            expected: 1,
            actual: 0,
        })
}

/// Post-process raw QKV projection outputs: deinterleave Q/gate and apply
/// per-head RMS norm. Usable from both the one-shot and persistent projection
/// paths.
pub(super) fn prepare_qkv_from_raw(
    attention: &Qwen35FullAttentionLayerPlan,
    q_full: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    sequence_length: usize,
    hidden_features: usize,
    rms_norm_eps: f32,
) -> Result<PreparedAttention, E2eError> {
    let query_features = checked_mul(attention.head_count, attention.head_dimension)?;

    let (q_values, q_gate) = deinterleave_q_gate(
        &q_full,
        sequence_length,
        attention.head_count,
        attention.head_dimension,
    )?;

    let q_values = per_head_rms_norm(
        &q_values,
        sequence_length,
        attention.head_count,
        attention.head_dimension,
        &attention.q_norm_values,
        rms_norm_eps,
    )?;
    let k_values = per_head_rms_norm(
        &k_proj,
        sequence_length,
        attention.kv_head_count,
        attention.head_dimension,
        &attention.k_norm_values,
        rms_norm_eps,
    )?;

    Ok(PreparedAttention {
        q_values,
        k_values,
        v_proj,
        q_gate,
        hidden_features,
        query_features,
    })
}

fn project_and_prepare_qkv(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    backend: Option<&Backend>,
) -> Result<PreparedAttention, E2eError> {
    let hidden_features = full_attention_hidden_features(attention)?;
    let expected_input_len = checked_mul(hidden_features, sequence_length)?;
    if input.len() != expected_input_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input_len,
            actual: input.len(),
        });
    }

    let query_features = checked_mul(attention.head_count, attention.head_dimension)?;
    let kv_features = checked_mul(attention.kv_head_count, attention.head_dimension)?;
    let query_features_x2 = checked_mul(query_features, 2)?;

    let qkv = if let Some(backend) = backend {
        project_qkv_graph(
            input,
            sequence_length,
            hidden_features,
            query_features_x2,
            kv_features,
            attention,
            backend,
        )?
    } else {
        let q_full = project_sequence(
            input,
            sequence_length,
            hidden_features,
            query_features_x2,
            &attention.q_weight_values,
        )?;
        let k_proj = project_sequence(
            input,
            sequence_length,
            hidden_features,
            kv_features,
            &attention.k_weight_values,
        )?;
        let v_proj = project_sequence(
            input,
            sequence_length,
            hidden_features,
            kv_features,
            &attention.v_weight_values,
        )?;
        QkvProjections {
            q_full,
            k_proj,
            v_proj,
        }
    };

    prepare_qkv_from_raw(
        attention,
        qkv.q_full,
        qkv.k_proj,
        qkv.v_proj,
        sequence_length,
        hidden_features,
        rms_norm_eps,
    )
}

fn qwen35_full_attention_core(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    attn_norm_weight: &[f32],
    state: Option<&mut Qwen35FullAttentionState>,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    fully_fused_attention_graph(
        attention,
        input,
        sequence_length,
        rms_norm_eps,
        attn_norm_weight,
        state,
        backend,
    )
}

/// De-interleave ggml's `[Q_h0, G_h0, Q_h1, G_h1, ...]` layout into
/// separate Q and gate buffers.
///
/// Both call sites (prefill multi-token and decode single-token) go through
/// this function so validation is unified.
fn deinterleave_q_gate(
    q_full: &[f32],
    sequence_length: usize,
    head_count: usize,
    head_dimension: usize,
) -> Result<(Vec<f32>, Vec<f32>), E2eError> {
    let query_features = checked_mul(head_count, head_dimension)?;
    let per_token_qg = checked_mul(query_features, 2)?;
    let expected_len = checked_mul(sequence_length, per_token_qg)?;
    if q_full.len() != expected_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_len,
            actual: q_full.len(),
        });
    }

    let total_out = checked_mul(sequence_length, query_features)?;
    let mut q_values = vec![0.0_f32; total_out];
    let mut q_gate = vec![0.0_f32; total_out];

    for ((src_token, q_dst_token), g_dst_token) in q_full
        .chunks_exact(per_token_qg)
        .zip(q_values.chunks_exact_mut(query_features))
        .zip(q_gate.chunks_exact_mut(query_features))
    {
        for head in 0..head_count {
            let hd = head_dimension;
            q_dst_token[head * hd..(head + 1) * hd]
                .copy_from_slice(&src_token[head * 2 * hd..head * 2 * hd + hd]);
            g_dst_token[head * hd..(head + 1) * hd]
                .copy_from_slice(&src_token[head * 2 * hd + hd..(head + 1) * 2 * hd]);
        }
    }

    Ok((q_values, q_gate))
}

/// NeoX-style RoPE configuration shared across Q and K rotations.
#[derive(Debug, Clone, Copy)]
pub(super) struct RopeParams {
    /// Number of dimensions to rotate (must be even, ≤ head_dimension).
    pub n_rot: usize,
    /// Base frequency for position encoding (e.g. 10000.0).
    pub freq_base: f32,
    /// Frequency scaling factor (typically 1.0).
    pub freq_scale: f32,
    /// Position offset for decode (0 for prefill, prompt_len for decode).
    pub position_offset: usize,
}

/// Apply NeoX-style rotary position embedding in-place.
///
/// For each token at position `rope.position_offset + pos`, rotates dimension pairs
/// `(x[k], x[k + n_rot/2])` for `k` in `0..n_rot/2` using angle
/// `theta_k = pos * freq_base^(-2k / n_rot)`.
/// Dimensions beyond `n_rot` are left unchanged.
pub(super) fn apply_neox_rope_in_place(
    values: &mut [f32],
    sequence_length: usize,
    head_count: usize,
    head_dimension: usize,
    rope: &RopeParams,
) -> Result<(), E2eError> {
    let total_features = checked_mul(head_count, head_dimension)?;
    let expected_len = checked_mul(sequence_length, total_features)?;
    if values.len() != expected_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_len,
            actual: values.len(),
        });
    }
    debug_assert!(rope.n_rot <= head_dimension && rope.n_rot.is_multiple_of(2));

    let half_rot = rope.n_rot / 2;
    let theta_scale = rope.freq_base.powf(-2.0 / rope.n_rot as f32);

    let cache_size = checked_mul(sequence_length, half_rot)?;
    let mut cos_cache = vec![0.0_f32; cache_size];
    let mut sin_cache = vec![0.0_f32; cache_size];
    for pos in 0..sequence_length {
        let mut theta = (rope.position_offset + pos) as f32;
        for k in 0..half_rot {
            let cache_idx = pos * half_rot + k;
            let angle = theta * rope.freq_scale;
            cos_cache[cache_idx] = angle.cos();
            sin_cache[cache_idx] = angle.sin();
            theta *= theta_scale;
        }
    }

    for pos in 0..sequence_length {
        let token_base = pos * total_features;
        for head in 0..head_count {
            let head_base = token_base + head * head_dimension;
            for k in 0..half_rot {
                let cache_idx = pos * half_rot + k;
                let cos_t = cos_cache[cache_idx];
                let sin_t = sin_cache[cache_idx];
                let idx0 = head_base + k;
                let idx1 = head_base + k + half_rot;
                let x0 = values[idx0];
                let x1 = values[idx1];
                values[idx0] = x0 * cos_t - x1 * sin_t;
                values[idx1] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
    Ok(())
}

/// Fully fused attention graph: layer norm + projection + deinterleave + norm + RoPE + scoring.
///
/// Accepts **un-normed** hidden state and applies layer pre-norm as the first
/// graph operation, eliminating the host↔device round-trip for normalization.
///
///   rms_norm(X, eps) * attn_norm_weight
///   → mul_mat(W_q/W_k/W_v, X_normed) → strided deinterleave Q/gate
///   → rms_norm + weight → rope_ext (NeoX mode=2)
///   → permute → cont → flash_attn_ext → sigmoid(gate) → mul
///   → reshape_2d → mul_mat(W_out)
///
/// When `state` is Some, reads back post-RoPE K and raw V for KV cache capture.
fn fully_fused_attention_graph(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    t: usize,
    rms_norm_eps: f32,
    attn_norm_weight: &[f32],
    state: Option<&mut Qwen35FullAttentionState>,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    use super::numeric::build_causal_mask_f16_bytes;
    use ggml_rs::{Dims, Length, RopeExtParams, Type};

    let dims = FullAttentionDims::new(attention)?;
    let FullAttentionDims {
        d,
        h,
        hkv,
        hidden,
        qf,
        qf2,
        kvf,
    } = dims;

    let expected_input = checked_mul(hidden, t)?;
    if input.len() != expected_input {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input,
            actual: input.len(),
        });
    }

    let ctx = Context::new_no_alloc_bytes(dims.estimate_memory(t))
        .map_err(|source| E2eError::ggml("Context::new(fully_fused_attn)", source))?;

    // --- Input tensors ---
    let x_raw = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden, t))
        .map_err(|source| E2eError::ggml("new<X>", source))?;

    // Layer pre-norm weight: [hidden_features]
    let attn_norm_w = ctx
        .new_tensor_1d::<f32>(Length::new(hidden))
        .map_err(|source| E2eError::ggml("new<attn_norm_w>", source))?;

    // In-graph layer pre-norm: rms_norm(X, eps) * attn_norm_weight
    let x_normed = ctx
        .rms_norm(&x_raw, rms_norm_eps)
        .map_err(|source| E2eError::ggml("rms_norm(X_layer)", source))?;
    let x = ctx
        .mul(&x_normed, &attn_norm_w)
        .map_err(|source| E2eError::ggml("mul(X_layer_norm)", source))?;

    // Weight tensors
    let w_q = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden, qf2))
        .map_err(|source| E2eError::ggml("new<W_q>", source))?;
    let w_k = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden, kvf))
        .map_err(|source| E2eError::ggml("new<W_k>", source))?;
    let w_v = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden, kvf))
        .map_err(|source| E2eError::ggml("new<W_v>", source))?;
    let w_out = ctx
        .new_tensor_2d::<f32>(Shape2D::new(qf, hidden))
        .map_err(|source| E2eError::ggml("new<W_out>", source))?;

    // Norm weight tensors: [D] each, broadcast across H and T.
    let q_norm_w = ctx
        .new_tensor_1d::<f32>(Length::new(d))
        .map_err(|source| E2eError::ggml("new<q_norm>", source))?;
    let k_norm_w = ctx
        .new_tensor_1d::<f32>(Length::new(d))
        .map_err(|source| E2eError::ggml("new<k_norm>", source))?;

    // Position tensor for RoPE: [T] i32
    let positions = ctx
        .new_tensor_1d::<i32>(Length::new(t))
        .map_err(|source| E2eError::ggml("new<positions>", source))?;

    // Causal mask as f16: [T, T, 1, 1]
    let mask = ctx
        .new_tensor(Type::F16, Dims::new([t, t, 1, 1]))
        .map_err(|source| E2eError::ggml("new<mask>", source))?;

    // --- QKV Projection ---
    let q_full = ctx
        .mul_mat(&w_q, &x)
        .map_err(|source| E2eError::ggml("mul_mat(Q)", source))?; // [qf2, T]
    let k_proj = ctx
        .mul_mat(&w_k, &x)
        .map_err(|source| E2eError::ggml("mul_mat(K)", source))?; // [kvf, T]
    let v_proj = ctx
        .mul_mat(&w_v, &x)
        .map_err(|source| E2eError::ggml("mul_mat(V)", source))?; // [kvf, T]

    // --- Deinterleave Q and Gate via strided view ---
    // Q_full layout per token: [Q_h0(D), G_h0(D), Q_h1(D), G_h1(D), ...]
    // Reshape to [D, 2*H, T] then take every-other slice along dim 1.
    let q_full_3d = ctx
        .reshape_3d(&q_full, d, 2 * h, t)
        .map_err(|source| E2eError::ggml("reshape_3d(Q_full)", source))?;

    let elem = std::mem::size_of::<f32>();
    let stride_h = 2 * d * elem; // skip one Q + one gate head
    let stride_t = 2 * h * d * elem; // full token

    // Q: even-indexed heads (offset 0)
    let q_view = ctx
        .view_3d(&q_full_3d, d, h, t, stride_h, stride_t, 0)
        .map_err(|source| E2eError::ggml("view_3d(Q)", source))?;
    let q_cont = ctx
        .cont(&q_view)
        .map_err(|source| E2eError::ggml("cont(Q_deinterleave)", source))?; // [D, H, T]

    // Gate: odd-indexed heads (offset D*sizeof(f32))
    let gate_view = ctx
        .view_3d(&q_full_3d, d, h, t, stride_h, stride_t, d * elem)
        .map_err(|source| E2eError::ggml("view_3d(Gate)", source))?;
    let gate_cont = ctx
        .cont(&gate_view)
        .map_err(|source| E2eError::ggml("cont(Gate_deinterleave)", source))?; // [D, H, T]

    // --- Per-head RMS norm ---
    // rms_norm normalizes along ne[0]=D for each (h, t) pair.
    let k_3d = ctx
        .reshape_3d(&k_proj, d, hkv, t)
        .map_err(|source| E2eError::ggml("reshape_3d(K)", source))?;

    let q_normed = ctx
        .rms_norm(&q_cont, rms_norm_eps)
        .map_err(|source| E2eError::ggml("rms_norm(Q)", source))?;
    let q_scaled = ctx
        .mul(&q_normed, &q_norm_w)
        .map_err(|source| E2eError::ggml("mul(Q_norm, weight)", source))?; // [D, H, T]

    let k_normed = ctx
        .rms_norm(&k_3d, rms_norm_eps)
        .map_err(|source| E2eError::ggml("rms_norm(K)", source))?;
    let k_scaled = ctx
        .mul(&k_normed, &k_norm_w)
        .map_err(|source| E2eError::ggml("mul(K_norm, weight)", source))?; // [D, Hkv, T]

    // --- NeoX RoPE (mode=2) ---
    let rope_params = RopeExtParams {
        n_dims: attention.rope_n_dims as i32,
        mode: 2, // NeoX: rotate first half, keep second half
        n_ctx_orig: 0,
        freq_base: attention.rope_freq_base,
        freq_scale: attention.rope_freq_scale,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 0.0,
        beta_slow: 0.0,
    };

    let q_rope = ctx
        .rope_ext_with_i32_positions(&q_scaled, &positions, None, rope_params)
        .map_err(|source| E2eError::ggml("rope_ext(Q)", source))?; // [D, H, T]
    let k_rope = ctx
        .rope_ext_with_i32_positions(&k_scaled, &positions, None, rope_params)
        .map_err(|source| E2eError::ggml("rope_ext(K)", source))?; // [D, Hkv, T]

    // --- Prepare for flash_attn_ext: [D, T, H, 1] ---
    let q_4d = ctx
        .reshape_4d(&q_rope, d, h, t, 1)
        .map_err(|source| E2eError::ggml("reshape_4d(Q)", source))?;
    let k_4d = ctx
        .reshape_4d(&k_rope, d, hkv, t, 1)
        .map_err(|source| E2eError::ggml("reshape_4d(K)", source))?;
    let v_3d = ctx
        .reshape_3d(&v_proj, d, hkv, t)
        .map_err(|source| E2eError::ggml("reshape_3d(V)", source))?;
    let v_4d = ctx
        .reshape_4d(&v_3d, d, hkv, t, 1)
        .map_err(|source| E2eError::ggml("reshape_4d(V)", source))?;

    // Permute [D, H, T, 1] → [D, T, H, 1] + cont for flash_attn_ext.
    let q_perm = ctx
        .permute(&q_4d, 0, 2, 1, 3)
        .map_err(|source| E2eError::ggml("permute(Q)", source))?;
    let k_perm = ctx
        .permute(&k_4d, 0, 2, 1, 3)
        .map_err(|source| E2eError::ggml("permute(K)", source))?;
    let v_perm = ctx
        .permute(&v_4d, 0, 2, 1, 3)
        .map_err(|source| E2eError::ggml("permute(V)", source))?;

    let q_c = ctx
        .cont(&q_perm)
        .map_err(|source| E2eError::ggml("cont(Q_perm)", source))?;
    let k_c = ctx
        .cont(&k_perm)
        .map_err(|source| E2eError::ggml("cont(K_perm)", source))?;
    let v_c = ctx
        .cont(&v_perm)
        .map_err(|source| E2eError::ggml("cont(V_perm)", source))?;

    // --- Flash attention + gating + output projection ---
    let attn = ctx
        .flash_attn_ext(
            &q_c,
            &k_c,
            &v_c,
            Some(&mask),
            attention.attention_scale,
            0.0,
            0.0,
        )
        .map_err(|source| E2eError::ggml("flash_attn_ext", source))?; // [D, H, T, 1]

    // Gate: reshape to 4D for sigmoid + element-wise mul with flash output.
    let gate_4d = ctx
        .reshape_4d(&gate_cont, d, h, t, 1)
        .map_err(|source| E2eError::ggml("reshape_4d(Gate)", source))?;
    let gate_sig = ctx
        .sigmoid(&gate_4d)
        .map_err(|source| E2eError::ggml("sigmoid(Gate)", source))?;
    let gated = ctx
        .mul(&attn, &gate_sig)
        .map_err(|source| E2eError::ggml("mul(attn, gate)", source))?; // [D, H, T, 1]

    // Output projection: [D, H, T, 1] → [H*D, T] → mul_mat(W_out) → [hidden, T]
    let gated_2d = ctx
        .reshape_2d(&gated, qf, t)
        .map_err(|source| E2eError::ggml("reshape_2d(gated)", source))?;
    let output = ctx
        .mul_mat(&w_out, &gated_2d)
        .map_err(|source| E2eError::ggml("mul_mat(output)", source))?; // [hidden, T]

    // --- Build, allocate, write, compute ---
    let mut graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("new_graph(fully_fused)", source))?;
    graph.build_forward_expand(&output);
    // Also include K_rope and V_proj in the graph for intermediate readback.
    graph.build_forward_expand(&k_rope);
    graph.build_forward_expand(&v_proj);

    let _buffer = ctx
        .allocate_tensors(backend)
        .map_err(|source| E2eError::ggml("allocate_tensors(fully_fused)", source))?;

    // Write input data (un-normed).
    x_raw
        .write_data_backend(input)
        .map_err(|source| E2eError::ggml("write<X>", source))?;

    // Layer pre-norm weight.
    attn_norm_w
        .write_data_backend(attn_norm_weight)
        .map_err(|source| E2eError::ggml("write<attn_norm_w>", source))?;

    // Write weight data.
    w_q.write_data_backend(&attention.q_weight_values)
        .map_err(|source| E2eError::ggml("write<W_q>", source))?;
    w_k.write_data_backend(&attention.k_weight_values)
        .map_err(|source| E2eError::ggml("write<W_k>", source))?;
    w_v.write_data_backend(&attention.v_weight_values)
        .map_err(|source| E2eError::ggml("write<W_v>", source))?;
    w_out
        .write_data_backend(&attention.output_weight_values)
        .map_err(|source| E2eError::ggml("write<W_out>", source))?;

    // Norm weights.
    q_norm_w
        .write_data_backend(&attention.q_norm_values)
        .map_err(|source| E2eError::ggml("write<q_norm>", source))?;
    k_norm_w
        .write_data_backend(&attention.k_norm_values)
        .map_err(|source| E2eError::ggml("write<k_norm>", source))?;

    // Position indices: [0, 1, 2, ..., T-1]
    let pos_data: Vec<i32> = (0..t as i32).collect();
    positions
        .write_data_backend(&pos_data)
        .map_err(|source| E2eError::ggml("write<positions>", source))?;

    // Causal mask as f16.
    let mask_bytes = build_causal_mask_f16_bytes(t);
    mask.write_bytes_backend(&mask_bytes)
        .map_err(|source| E2eError::ggml("write<mask>", source))?;

    backend
        .compute(&mut graph)
        .map_err(|source| E2eError::ggml("compute(fully_fused)", source))?;

    // Read back post-RoPE K and raw V for KV cache state capture.
    if let Some(state) = state {
        let k_rope_data: Vec<f32> = k_rope
            .read_data_backend()
            .map_err(|source| E2eError::ggml("read<K_rope>", source))?;
        let v_data: Vec<f32> = v_proj
            .read_data_backend()
            .map_err(|source| E2eError::ggml("read<V_proj>", source))?;
        state.append_batch(&k_rope_data, &v_data, t)?;
    }

    output
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read<output>", source))
}
/// GPU-accelerated attention scoring via `flash_attn_ext`.
///
/// Builds a temporary ggml graph each decode step that:
/// 1. Uploads Q, live KV cache prefix, and gate to the backend
/// 2. Permutes K/V from host cache layout `[D, Hkv, T]` → flash `[D, T, Hkv]`
/// 3. Runs `flash_attn_ext` (Q·K scoring + softmax + V aggregation)
/// 4. Applies sigmoid gating
///
/// Returns the gated head outputs matching host scoring loop semantics.
/// On any failure the caller should fall back to the host scoring loop.
fn decode_scoring_gpu(
    q_values: &[f32],
    q_gate: &[f32],
    state: &Qwen35FullAttentionState,
    total_tokens: usize,
    attention: &Qwen35FullAttentionLayerPlan,
    query_features: usize,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    let d = attention.head_dimension;
    let h = attention.head_count;
    let hkv = attention.kv_head_count;
    let t = total_tokens;

    // Metadata-only context (backend manages data); 2 MB is generous for ~10 tensors + 1 graph.
    let ctx = Context::new_no_alloc(2 * 1024 * 1024)
        .map_err(|source| E2eError::ggml("ctx(scoring_gpu)", source))?;

    // Input tensors — shapes follow flash_attn_ext convention.
    // Q: [D, T_q=1, H, 1]
    let q = ctx
        .new_tensor_4d::<f32>(Shape4D::new(d, 1, h, 1))
        .map_err(|source| E2eError::ggml("q_tensor", source))?;
    // K/V in host cache layout [D, Hkv, T, 1]; will be permuted to [D, T, Hkv, 1].
    let k_raw = ctx
        .new_tensor_4d::<f32>(Shape4D::new(d, hkv, t, 1))
        .map_err(|source| E2eError::ggml("k_tensor", source))?;
    let v_raw = ctx
        .new_tensor_4d::<f32>(Shape4D::new(d, hkv, t, 1))
        .map_err(|source| E2eError::ggml("v_tensor", source))?;
    // Gate: [D, H, 1, 1] — matches flash_attn_ext output shape.
    let gate = ctx
        .new_tensor_4d::<f32>(Shape4D::new(d, h, 1, 1))
        .map_err(|source| E2eError::ggml("gate_tensor", source))?;

    // Permute K/V from [D, Hkv, T, 1] → [D, T, Hkv, 1] + make contiguous.
    let k_perm = ctx
        .permute(&k_raw, 0, 2, 1, 3)
        .map_err(|source| E2eError::ggml("permute(K)", source))?;
    let k = ctx
        .cont(&k_perm)
        .map_err(|source| E2eError::ggml("cont(K)", source))?;
    let v_perm = ctx
        .permute(&v_raw, 0, 2, 1, 3)
        .map_err(|source| E2eError::ggml("permute(V)", source))?;
    let v = ctx
        .cont(&v_perm)
        .map_err(|source| E2eError::ggml("cont(V)", source))?;

    // flash_attn_ext: Q·K scoring + softmax + V aggregation.
    // No causal mask for single-query decode (token attends to itself + all past).
    let attn = ctx
        .flash_attn_ext(&q, &k, &v, None, attention.attention_scale, 0.0, 0.0)
        .map_err(|source| E2eError::ggml("flash_attn_ext(decode)", source))?;
    // Output: [D, H, 1, 1]

    // Gating: sigmoid(gate) ⊙ attn
    let gate_sig = ctx
        .sigmoid(&gate)
        .map_err(|source| E2eError::ggml("sigmoid(gate)", source))?;
    let gated = ctx
        .mul(&attn, &gate_sig)
        .map_err(|source| E2eError::ggml("mul(attn,gate)", source))?;

    // Build graph → allocate → upload → compute → readback.
    let mut graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("new_graph(scoring)", source))?;
    graph.build_forward_expand(&gated);

    let _buffer = ctx
        .allocate_tensors(backend)
        .map_err(|source| E2eError::ggml("allocate(scoring)", source))?;

    let kv_prefix_len = t * state.kv_features;
    q.write_data_backend(q_values)
        .map_err(|source| E2eError::ggml("write(Q)", source))?;
    k_raw
        .write_data_backend(&state.k_cache[..kv_prefix_len])
        .map_err(|source| E2eError::ggml("write(K)", source))?;
    v_raw
        .write_data_backend(&state.v_cache[..kv_prefix_len])
        .map_err(|source| E2eError::ggml("write(V)", source))?;
    gate.write_data_backend(q_gate)
        .map_err(|source| E2eError::ggml("write(gate)", source))?;

    backend
        .compute(&mut graph)
        .map_err(|source| E2eError::ggml("compute(scoring)", source))?;

    let outputs = gated
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read(gated)", source))?;

    if outputs.len() != query_features {
        return Err(E2eError::BufferLengthMismatch {
            expected: query_features,
            actual: outputs.len(),
        });
    }

    Ok(outputs)
}

/// Core full attention decode logic: RoPE → KV cache append → scoring → gating.
///
/// Takes prepared QKV projections and returns gated head outputs (before output
/// projection). The caller is responsible for projecting the output.
///
/// When `persistent_kv` is `Some`, uses persistent backend-resident KV cache
/// (O(1) per-step upload). Otherwise falls back to `decode_scoring_gpu` (O(T)
/// upload) or host scoring loop.
pub(super) fn full_attention_decode_core(
    prepared: PreparedAttention,
    attention: &Qwen35FullAttentionLayerPlan,
    state: &mut Qwen35FullAttentionState,
    backend: Option<&Backend>,
    persistent_kv: Option<&PersistentKvCache<'static>>,
    scoring_ctx: Option<&mut PersistentScoringContext>,
) -> Result<Vec<f32>, E2eError> {
    let PreparedAttention {
        mut q_values,
        mut k_values,
        v_proj,
        q_gate,
        hidden_features: _,
        query_features,
    } = prepared;

    if attention.kv_head_count == 0 || !attention.head_count.is_multiple_of(attention.kv_head_count)
    {
        return Err(E2eError::BufferLengthMismatch {
            expected: attention.head_count,
            actual: attention.kv_head_count,
        });
    }

    let hd = attention.head_dimension;

    let rope = RopeParams {
        n_rot: attention.rope_n_dims,
        freq_base: attention.rope_freq_base,
        freq_scale: attention.rope_freq_scale,
        position_offset: state.token_count(),
    };

    apply_neox_rope_in_place(
        &mut q_values,
        1,
        attention.head_count,
        attention.head_dimension,
        &rope,
    )?;
    apply_neox_rope_in_place(
        &mut k_values,
        1,
        attention.kv_head_count,
        attention.head_dimension,
        &rope,
    )?;

    state.append_batch(&k_values, &v_proj, 1)?;
    let total_tokens = state.token_count();

    // Also append to persistent backend KV cache (O(1) upload).
    if let Some(kv) = persistent_kv {
        // cached_len was incremented by append_batch, so the new token is at
        // position `total_tokens - 1`.
        let _ = kv.append_token(&k_values, &v_proj, total_tokens - 1);
    }

    // Try persistent GPU scoring first (O(1) upload, O(T) on-device permute).
    if let (Some(kv), Some(backend)) = (persistent_kv, backend)
        && !state.gpu_scoring_failed
    {
        if let Ok(outputs) = decode_scoring_gpu_persistent(
            &q_values,
            &q_gate,
            total_tokens,
            attention,
            kv,
            backend,
            scoring_ctx,
        ) {
            return Ok(outputs);
        }
        state.gpu_scoring_failed = true;
    }

    // Fallback: try ephemeral GPU scoring (O(T) upload).
    if let Some(backend) = backend
        && !state.gpu_scoring_failed
    {
        if let Ok(outputs) = decode_scoring_gpu(
            &q_values,
            &q_gate,
            state,
            total_tokens,
            attention,
            query_features,
            backend,
        ) {
            return Ok(outputs);
        }
        // First GPU failure — disable for future decode steps.
        state.gpu_scoring_failed = true;
    }

    let groups = attention.head_count / attention.kv_head_count;
    let mut head_outputs = vec![0.0_f32; query_features];
    for head in 0..attention.head_count {
        let kv_head = head / groups;
        let q = &q_values[head * hd..(head + 1) * hd];

        let mut scores = vec![f32::NEG_INFINITY; total_tokens];
        for (source, score) in scores.iter_mut().enumerate().take(total_tokens) {
            let k = state.k_head_at(source, kv_head, hd);
            *score = dot(q, k) * attention.attention_scale;
        }
        let weights = softmax_prefix(&scores, total_tokens);

        let dst = &mut head_outputs[head * hd..(head + 1) * hd];
        for (source, weight) in weights.iter().copied().enumerate() {
            let v = state.v_head_at(source, kv_head, hd);
            for index in 0..hd {
                dst[index] += v[index] * weight;
            }
        }

        let gate = &q_gate[head * hd..(head + 1) * hd];
        for index in 0..hd {
            dst[index] *= sigmoid_scalar(gate[index]);
        }
    }

    Ok(head_outputs)
}

///
/// Processes one token using the KV cache accumulated during prefill (and
/// previous decode steps). The new K/V are appended to `state` BEFORE attention
/// so the token attends to itself.
pub(super) fn qwen35_full_attention_decode_step(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    rms_norm_eps: f32,
    state: &mut Qwen35FullAttentionState,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    let prepared = project_and_prepare_qkv(attention, input, 1, rms_norm_eps, Some(backend))?;
    let hidden_features = prepared.hidden_features;
    let query_features = prepared.query_features;
    let head_outputs =
        full_attention_decode_core(prepared, attention, state, Some(backend), None, None)?;

    project_sequence_graph(
        &head_outputs,
        1,
        query_features,
        hidden_features,
        &attention.output_weight_values,
        backend,
    )
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen35_full_attention_qgate_split_is_head_interleaved() {
        let q_full: Vec<f32> = vec![
            1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 40.0, 50.0, 60.0,
        ];
        let (q_values, q_gate) = deinterleave_q_gate(&q_full, 1, 2, 3).unwrap();
        assert_eq!(q_values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(q_gate, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn qwen35_full_attention_qgate_split_multi_token() {
        let q_full: Vec<f32> = vec![
            1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0, 5.0, 6.0, 50.0, 60.0, 7.0, 8.0, 70.0, 80.0,
        ];
        let (q_values, q_gate) = deinterleave_q_gate(&q_full, 2, 2, 2).unwrap();
        assert_eq!(q_values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(q_gate, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    }

    #[test]
    fn rope_identity_at_position_zero() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let original = values.clone();
        apply_neox_rope_in_place(
            &mut values,
            1,
            1,
            4,
            &RopeParams {
                n_rot: 4,
                freq_base: 10000.0,
                freq_scale: 1.0,
                position_offset: 0,
            },
        )
        .unwrap();
        for (a, b) in values.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
        }
    }

    #[test]
    fn rope_rotates_at_nonzero_position() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 0.0];
        apply_neox_rope_in_place(
            &mut values,
            2,
            1,
            4,
            &RopeParams {
                n_rot: 4,
                freq_base: 1.0,
                freq_scale: 1.0,
                position_offset: 0,
            },
        )
        .unwrap();

        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[1] - 2.0).abs() < 1e-6);
        assert!((values[2] - 3.0).abs() < 1e-6);
        assert!((values[3] - 4.0).abs() < 1e-6);

        let cos1 = 1.0_f32.cos();
        let sin1 = 1.0_f32.sin();
        assert!(
            (values[4] - cos1).abs() < 1e-6,
            "expected {cos1}, got {}",
            values[4]
        );
        assert!((values[5]).abs() < 1e-6);
        assert!(
            (values[6] - sin1).abs() < 1e-6,
            "expected {sin1}, got {}",
            values[6]
        );
        assert!((values[7]).abs() < 1e-6);
    }

    #[test]
    fn rope_preserves_dims_beyond_n_rot() {
        let mut values = [
            1.0_f32, 2.0, 3.0, 4.0, 99.0, 88.0, 1.0, 2.0, 3.0, 4.0, 99.0, 88.0,
        ];
        apply_neox_rope_in_place(
            &mut values,
            2,
            1,
            6,
            &RopeParams {
                n_rot: 4,
                freq_base: 10000.0,
                freq_scale: 1.0,
                position_offset: 0,
            },
        )
        .unwrap();
        assert!((values[4] - 99.0).abs() < 1e-6);
        assert!((values[5] - 88.0).abs() < 1e-6);
        assert!((values[10] - 99.0).abs() < 1e-6);
        assert!((values[11] - 88.0).abs() < 1e-6);
    }

    #[test]
    fn rope_multi_head_applies_same_rotation_per_head() {
        let mut buf = [
            0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        apply_neox_rope_in_place(
            &mut buf,
            2,
            2,
            4,
            &RopeParams {
                n_rot: 4,
                freq_base: 1.0,
                freq_scale: 1.0,
                position_offset: 0,
            },
        )
        .unwrap();
        assert_eq!(&buf[8..12], &buf[12..16]);
    }

    #[test]
    fn rope_position_offset_matches_sequential() {
        // RoPE at offset=2 for 1 token should match position 2 from a 3-token batch.
        let mut batch = vec![0.0_f32; 3 * 4]; // 3 tokens, hd=4
        batch[2 * 4] = 1.0;
        batch[2 * 4 + 1] = 2.0;
        batch[2 * 4 + 2] = 3.0;
        batch[2 * 4 + 3] = 4.0;
        apply_neox_rope_in_place(
            &mut batch,
            3,
            1,
            4,
            &RopeParams {
                n_rot: 4,
                freq_base: 10000.0,
                freq_scale: 1.0,
                position_offset: 0,
            },
        )
        .unwrap();

        let mut single = vec![1.0, 2.0, 3.0, 4.0];
        apply_neox_rope_in_place(
            &mut single,
            1,
            1,
            4,
            &RopeParams {
                n_rot: 4,
                freq_base: 10000.0,
                freq_scale: 1.0,
                position_offset: 2,
            },
        )
        .unwrap();

        for (i, (a, b)) in single.iter().zip(&batch[8..12]).enumerate() {
            assert!((a - b).abs() < 1e-6, "dim {i}: offset={a} vs batch={b}");
        }
    }

    #[test]
    fn full_attention_prefill_then_decode_matches_full_reprocess() {
        // Build a small deterministic plan: 2 heads, 1 kv_head (GQA), hd=4.
        let head_count = 2;
        let kv_head_count = 1;
        let hd = 4;
        let query_features = head_count * hd; // 8
        let kv_features = kv_head_count * hd; // 4
        let hidden = 6;

        // Q weight: hidden → query_features*2 (Q+Gate interleaved)
        let q_weight: Vec<f32> = (0..hidden * query_features * 2)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let k_weight: Vec<f32> = (0..hidden * kv_features)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.08)
            .collect();
        let v_weight: Vec<f32> = (0..hidden * kv_features)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
            .collect();
        let output_weight: Vec<f32> = (0..query_features * hidden)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
            .collect();
        let q_norm = vec![1.0_f32; hd];
        let k_norm = vec![1.0_f32; hd];

        let plan = super::super::plan::Qwen35FullAttentionLayerPlan {
            norm_values: vec![1.0; hidden],
            q_norm_values: q_norm,
            k_norm_values: k_norm,
            q_weight_values: q_weight,
            k_weight_values: k_weight,
            v_weight_values: v_weight,
            output_weight_values: output_weight,
            head_count,
            kv_head_count,
            head_dimension: hd,
            attention_scale: 1.0 / (hd as f32).sqrt(),
            rope_n_dims: hd,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
        };

        // 3-token prompt + 1 decode token.
        let prompt: Vec<f32> = (0..3 * hidden).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let new_token: Vec<f32> = (0..hidden).map(|i| (i as f32 + 50.0) * 0.05).collect();

        crate::backend::ensure_backends_loaded();
        let backend =
            Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend should be available");

        // Full reprocess: 4 tokens at once.
        let full_input: Vec<f32> = prompt.iter().chain(new_token.iter()).copied().collect();
        let norm_weight = &plan.norm_values;
        let full_output =
            qwen35_full_attention_inference(&plan, &full_input, 4, 1e-5, norm_weight, &backend)
                .unwrap();
        let expected = &full_output[3 * hidden..4 * hidden];

        // Prefill 3 tokens, then decode 1.
        let mut state = Qwen35FullAttentionState::new(4, kv_head_count, hd).unwrap();
        let _prefill_out = qwen35_full_attention_prefill(
            &plan,
            &prompt,
            3,
            1e-5,
            norm_weight,
            &mut state,
            &backend,
        )
        .unwrap();

        // Decode path: apply host-side rms_norm + weight to match the in-graph
        // norm that inference/prefill now perform.
        let normalized_token = super::super::tensor_ops::rms_norm_with_weight(
            &new_token,
            hidden,
            1,
            norm_weight,
            1e-5,
        )
        .unwrap();
        let decode_out =
            qwen35_full_attention_decode_step(&plan, &normalized_token, 1e-5, &mut state, &backend)
                .unwrap();

        for (i, (a, b)) in decode_out.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "feature {i}: decode={a} vs full={b}, diff={}",
                (a - b).abs()
            );
        }
    }

    /// Verifies that `decode_scoring_gpu` (flash_attn_ext path) produces the
    /// same gated head outputs as the host scoring loop inside
    /// `full_attention_decode_core`.
    ///
    /// Uses GQA (H > Hkv) with multiple cached tokens (Tkv > 1) to exercise
    /// the KV cache permutation and grouped-query layout.
    #[test]
    fn gpu_scoring_matches_host_scoring() {
        let head_count = 4;
        let kv_head_count = 2; // GQA: 2 groups
        let hd = 4;
        let query_features = head_count * hd; // 16
        let kv_features = kv_head_count * hd; // 8
        let hidden = 6;

        // Deterministic weights.
        let q_weight: Vec<f32> = (0..hidden * query_features * 2)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let k_weight: Vec<f32> = (0..hidden * kv_features)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.08)
            .collect();
        let v_weight: Vec<f32> = (0..hidden * kv_features)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
            .collect();
        let output_weight: Vec<f32> = (0..query_features * hidden)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
            .collect();
        let q_norm = vec![1.0_f32; hd];
        let k_norm = vec![1.0_f32; hd];

        let plan = super::super::plan::Qwen35FullAttentionLayerPlan {
            norm_values: vec![1.0; hidden],
            q_norm_values: q_norm,
            k_norm_values: k_norm,
            q_weight_values: q_weight,
            k_weight_values: k_weight,
            v_weight_values: v_weight,
            output_weight_values: output_weight,
            head_count,
            kv_head_count,
            head_dimension: hd,
            attention_scale: 1.0 / (hd as f32).sqrt(),
            rope_n_dims: hd,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
        };

        // 5-token prompt + 1 decode token — ensures Tkv > 1 for cache.
        let prompt: Vec<f32> = (0..5 * hidden).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let decode_token: Vec<f32> = (0..hidden).map(|i| (i as f32 + 50.0) * 0.05).collect();

        crate::backend::ensure_backends_loaded();
        let backend =
            Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend should be available");

        // Prefill to populate KV cache with 5 tokens.
        let norm_weight = &plan.norm_values;
        let mut state_host = Qwen35FullAttentionState::new(6, kv_head_count, hd).unwrap();
        let _prefill = qwen35_full_attention_prefill(
            &plan,
            &prompt,
            5,
            1e-5,
            norm_weight,
            &mut state_host,
            &backend,
        )
        .unwrap();

        // Clone state so both paths start from the same KV cache snapshot.
        let mut state_gpu = state_host.clone();

        // Project decode token (shared for both paths).
        let normalized = super::super::tensor_ops::rms_norm_with_weight(
            &decode_token,
            hidden,
            1,
            norm_weight,
            1e-5,
        )
        .unwrap();

        // Host scoring: backend = None.
        let prepared_host =
            project_and_prepare_qkv(&plan, &normalized, 1, 1e-5, Some(&backend)).unwrap();
        let host_out =
            full_attention_decode_core(prepared_host, &plan, &mut state_host, None, None, None)
                .unwrap();

        // GPU scoring: backend = Some.
        let prepared_gpu =
            project_and_prepare_qkv(&plan, &normalized, 1, 1e-5, Some(&backend)).unwrap();
        let gpu_out = full_attention_decode_core(
            prepared_gpu,
            &plan,
            &mut state_gpu,
            Some(&backend),
            None,
            None,
        )
        .unwrap();

        assert_eq!(host_out.len(), gpu_out.len());
        for (i, (h, g)) in host_out.iter().zip(gpu_out.iter()).enumerate() {
            assert!(
                (h - g).abs() < 1e-4,
                "head_output[{i}]: host={h} vs gpu={g}, diff={}",
                (h - g).abs()
            );
        }
    }
}
