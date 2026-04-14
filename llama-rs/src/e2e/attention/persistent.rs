//! Backend-resident persistent KV cache and GPU-accelerated scoring.

use crate::e2e::error::E2eError;
use crate::e2e::plan::Qwen35FullAttentionLayerPlan;
use crate::e2e::tensor_ops::upload_weight;
use ggml_rs::{Backend, BackendBuffer, Context, GraphAllocator, Shape4D, Tensor};

// ---------------------------------------------------------------------------
// PersistentKvCache
// ---------------------------------------------------------------------------

/// Backend-resident KV cache for GPU-accelerated attention scoring.
///
/// Pre-allocates max-size K/V tensors on the backend. Each decode step:
/// - Appends only the new token's K/V via `write_data_backend_at` (O(1))
/// - Ephemeral scoring graph creates `view_4d_of` into these tensors
///
/// Host `Vec<f32>` KV cache in `Qwen35FullAttentionState` remains the
/// source of truth (for checkpoint serialization and host-scoring fallback).
/// This struct is runtime-only decode infrastructure — not serializable.
pub(in crate::e2e) struct PersistentKvCache<'ctx> {
    /// K values: `[D, MaxT, Hkv, 1]` on backend (flash-friendly time-major layout).
    k_tensor: Tensor<'ctx, f32>,
    /// V values: `[D, MaxT, Hkv, 1]` on backend (flash-friendly time-major layout).
    v_tensor: Tensor<'ctx, f32>,
    /// Backend buffer keeping tensors alive on device.
    _buffer: BackendBuffer<'ctx>,
    /// Head dimension (D).
    pub(super) head_dim: usize,
    /// Number of KV heads (Hkv).
    pub(super) kv_head_count: usize,
    /// Max tokens this cache can hold.
    pub(super) max_tokens: usize,
}

impl<'ctx> PersistentKvCache<'ctx> {
    /// Append one token's K/V to the backend tensors at the given position.
    ///
    /// `cached_len` is the position index (0-based) of the new token. The
    /// caller must ensure `cached_len < max_tokens`.
    ///
    /// Layout is `[D, MaxT, Hkv, 1]` (flash-friendly time-major). Each KV
    /// head's D-element slice is written to the appropriate stride offset.
    pub(in crate::e2e) fn append_token(
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
    pub(in crate::e2e) fn seed_from_host(
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
/// Returns `(PersistentKvCache<'static>, Context)`. The handle is first so
/// that even if stored as a single value, it drops before the context.
/// The context must be stored in a parallel container and outlive the cache
/// handles (same pattern as `PersistentDecodeProjection`).
pub(in crate::e2e) fn build_persistent_kv_cache(
    attention: &Qwen35FullAttentionLayerPlan,
    max_tokens: usize,
    backend: &Backend,
) -> Result<(PersistentKvCache<'static>, Context), E2eError> {
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

    Ok((cache, ctx))
}

// ---------------------------------------------------------------------------
// PersistentScoringContext
// ---------------------------------------------------------------------------

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
pub(in crate::e2e) struct PersistentScoringContext {
    gallocr: GraphAllocator,
}

impl PersistentScoringContext {
    /// Build a persistent scoring context by reserving a buffer for the
    /// worst-case (max_tokens) scoring graph topology.
    pub(in crate::e2e) fn new(
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

// ---------------------------------------------------------------------------
// Scoring graph (private)
// ---------------------------------------------------------------------------

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
pub(super) fn decode_scoring_gpu_persistent(
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
    upload_weight(&sg.q, q_values, "write(Q)")?;
    upload_weight(&sg.gate, q_gate, "write(gate)")?;

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
