use super::*;

/// Resolves and decodes reusable attention weights for one transformer layer.
pub fn resolve_attention_weights_for_layer(
    model: &GgufModel,
    layer: usize,
    config: AttentionInferenceConfig,
) -> Result<AttentionWeights, InferenceError> {
    let layer_names = resolve_llama_layer_tensor_names(model, layer)
        .map_err(|source| InferenceError::naming("resolve_llama_layer_tensor_names", source))?;
    AttentionWeights::from_model_layer(model, &layer_names, config)
}

/// Resolves attention weights with auto-derived topology and optional RoPE metadata.
pub fn resolve_attention_weights_for_layer_auto(
    model: &GgufModel,
    layer: usize,
    sequence_length: usize,
) -> Result<AttentionWeights, InferenceError> {
    let dimensions = resolve_llama_layer_dimensions(model, layer)?;
    let config = AttentionInferenceConfig::from_layer_dimensions(dimensions, sequence_length)?;
    resolve_attention_weights_for_layer(model, layer, config)
}

/// Runs minimal self-attention from resolved GGUF layer tensors.
pub fn attention_inference_for_layer(
    model: &GgufModel,
    layer: usize,
    input: &[f32],
    config: AttentionInferenceConfig,
    backend_kind: LlamaBackend,
) -> Result<AttentionInferenceReport, InferenceError> {
    attention_inference_for_layer_repeats(model, layer, input, config, backend_kind, 1)
}

/// Runs minimal self-attention from resolved GGUF layer tensors with explicit repeats.
pub fn attention_inference_for_layer_repeats(
    model: &GgufModel,
    layer: usize,
    input: &[f32],
    config: AttentionInferenceConfig,
    backend_kind: LlamaBackend,
    repeats: usize,
) -> Result<AttentionInferenceReport, InferenceError> {
    let weights = resolve_attention_weights_for_layer(model, layer, config)?;
    attention_inference_with_weights_repeats(&weights, input, backend_kind, repeats)
}

/// Runs minimal self-attention with auto-derived topology and RoPE metadata.
pub fn attention_inference_for_layer_auto(
    model: &GgufModel,
    layer: usize,
    input: &[f32],
    sequence_length: usize,
    backend_kind: LlamaBackend,
) -> Result<AttentionInferenceReport, InferenceError> {
    attention_inference_for_layer_auto_repeats(
        model,
        layer,
        input,
        sequence_length,
        backend_kind,
        1,
    )
}

/// Runs minimal self-attention with auto-derived topology and explicit repeats.
pub fn attention_inference_for_layer_auto_repeats(
    model: &GgufModel,
    layer: usize,
    input: &[f32],
    sequence_length: usize,
    backend_kind: LlamaBackend,
    repeats: usize,
) -> Result<AttentionInferenceReport, InferenceError> {
    let weights = resolve_attention_weights_for_layer_auto(model, layer, sequence_length)?;
    attention_inference_with_weights_repeats(&weights, input, backend_kind, repeats)
}

/// Runs minimal self-attention with reusable decoded weights.
pub fn attention_inference_with_weights(
    weights: &AttentionWeights,
    input: &[f32],
    backend_kind: LlamaBackend,
) -> Result<AttentionInferenceReport, InferenceError> {
    attention_inference_with_weights_repeats(weights, input, backend_kind, 1)
}

/// Runs minimal self-attention with reusable decoded weights and explicit repeats.
pub fn attention_inference_with_weights_repeats(
    weights: &AttentionWeights,
    input: &[f32],
    backend_kind: LlamaBackend,
    repeats: usize,
) -> Result<AttentionInferenceReport, InferenceError> {
    let config = weights.config;
    let hidden_features = config.hidden_features();
    let sequence_length = config.sequence_length();
    let expected_input_len = hidden_features
        .checked_mul(sequence_length)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    if input.len() != expected_input_len {
        return Err(InferenceError::InvalidInputLength {
            expected: expected_input_len,
            actual: input.len(),
        });
    }
    if repeats == 0 {
        return Err(InferenceError::InvalidRepeats);
    }

    let ctx_size = recommended_attention_backend_memory_bytes(config)?;
    let runtime = DefaultBackendRuntimeBuilder.build_runtime(backend_kind, ctx_size)?;
    let backend = runtime.backend;
    let backend_name = runtime.backend_name;
    let ctx = runtime.ctx;

    let query_features = config.query_features();
    let kv_features = config.kv_features();

    let w_q = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, query_features))
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_Q>", source))?;
    let w_k = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, kv_features))
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_K>", source))?;
    let w_v = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, kv_features))
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_V>", source))?;
    let w_o = ctx
        .new_tensor_2d::<f32>(Shape2D::new(query_features, hidden_features))
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_O>", source))?;
    let x = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, sequence_length))
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<X>", source))?;

    let q = ctx
        .mul_mat(&w_q, &x)
        .map_err(|source| InferenceError::ggml("Context::mul_mat(Q)", source))?;
    let k = ctx
        .mul_mat(&w_k, &x)
        .map_err(|source| InferenceError::ggml("Context::mul_mat(K)", source))?;
    let v = ctx
        .mul_mat(&w_v, &x)
        .map_err(|source| InferenceError::ggml("Context::mul_mat(V)", source))?;

    let positions = if matches!(config.rotary, RotaryEmbedding::Llama(_)) {
        Some(
            ctx.new_tensor_1d::<i32>(Length::new(sequence_length))
                .map_err(|source| InferenceError::ggml("Context::new_i32_tensor_1d_len", source))?,
        )
    } else {
        None
    };
    let mask = if matches!(config.mask, AttentionMaskPolicy::Causal { .. }) {
        Some(
            ctx.new_tensor_2d::<f32>(Shape2D::new(sequence_length, sequence_length))
                .map_err(|source| {
                    InferenceError::ggml("Context::new_f32_tensor_2d_shape<CAUSAL_MASK>", source)
                })?,
        )
    } else {
        None
    };

    let mut output_projection = None;
    let bytes_per_element = std::mem::size_of::<f32>();
    let q_row_stride = query_features
        .checked_mul(bytes_per_element)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    let kv_row_stride = kv_features
        .checked_mul(bytes_per_element)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    let o_row_stride = q_row_stride;
    let attention_scale = 1.0 / (config.head_dimension() as f32).sqrt();
    let rotary_applier = LlamaRotaryApplier;

    for head in 0..config.query_head_count() {
        let query_offset = head
            .checked_mul(config.head_dimension())
            .and_then(|value| value.checked_mul(bytes_per_element))
            .ok_or(InferenceError::MemorySizeOverflow)?;
        let kv_head = head / config.layout.kv_group_size();
        let kv_offset = kv_head
            .checked_mul(config.head_dimension())
            .and_then(|value| value.checked_mul(bytes_per_element))
            .ok_or(InferenceError::MemorySizeOverflow)?;

        let q_head = ctx
            .view_2d(
                &q,
                config.head_dimension(),
                sequence_length,
                q_row_stride,
                query_offset,
            )
            .map_err(|source| InferenceError::ggml("Context::view_2d(Q_HEAD)", source))?;
        let k_head = ctx
            .view_2d(
                &k,
                config.head_dimension(),
                sequence_length,
                kv_row_stride,
                kv_offset,
            )
            .map_err(|source| InferenceError::ggml("Context::view_2d(K_HEAD)", source))?;
        let v_head = ctx
            .view_2d(
                &v,
                config.head_dimension(),
                sequence_length,
                kv_row_stride,
                kv_offset,
            )
            .map_err(|source| InferenceError::ggml("Context::view_2d(V_HEAD)", source))?;

        let q_head = rotary_applier.apply_single_with_sequence(
            &ctx,
            &q_head,
            positions.as_ref(),
            config,
            sequence_length,
        )?;
        let k_head = rotary_applier.apply_single_with_sequence(
            &ctx,
            &k_head,
            positions.as_ref(),
            config,
            sequence_length,
        )?;

        let scores = ctx
            .mul_mat(&k_head, &q_head)
            .map_err(|source| InferenceError::ggml("Context::mul_mat(K_HEAD*Q_HEAD)", source))?;

        let probabilities = ctx
            .soft_max_ext(&scores, mask.as_ref(), attention_scale, 0.0)
            .map_err(|source| InferenceError::ggml("Context::soft_max_ext", source))?;
        let v_t = ctx
            .transpose(&v_head)
            .map_err(|source| InferenceError::ggml("Context::transpose(V_HEAD)", source))?;
        let v_t = ctx
            .cont(&v_t)
            .map_err(|source| InferenceError::ggml("Context::cont(V_HEAD)", source))?;
        let head_output = ctx
            .mul_mat(&v_t, &probabilities)
            .map_err(|source| InferenceError::ggml("Context::mul_mat(VT*P)", source))?;
        let w_o_head = ctx
            .view_2d(
                &w_o,
                config.head_dimension(),
                hidden_features,
                o_row_stride,
                query_offset,
            )
            .map_err(|source| InferenceError::ggml("Context::view_2d(W_O_HEAD)", source))?;
        let projected = ctx
            .mul_mat(&w_o_head, &head_output)
            .map_err(|source| InferenceError::ggml("Context::mul_mat(W_O_HEAD*HEAD)", source))?;

        output_projection = Some(if let Some(acc) = output_projection {
            ctx.add(&acc, &projected)
                .map_err(|source| InferenceError::ggml("Context::add(head_acc)", source))?
        } else {
            projected
        });
    }

    let y = output_projection.ok_or(InferenceError::InvalidAttentionLayout {
        hidden_features,
        query_head_count: config.query_head_count(),
        kv_head_count: config.kv_head_count(),
    })?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| InferenceError::ggml("Context::new_graph", source))?;
    graph.build_forward_expand(&y);
    let _buffer = ctx
        .allocate_tensors(&backend)
        .map_err(|source| InferenceError::ggml("Context::allocate_tensors", source))?;

    w_q.write_data_backend(weights.q_values())
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<W_Q>", source))?;
    w_k.write_data_backend(weights.k_values())
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<W_K>", source))?;
    w_v.write_data_backend(weights.v_values())
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<W_V>", source))?;
    w_o.write_data_backend(weights.o_values())
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<W_O>", source))?;
    x.write_data_backend(input)
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<X>", source))?;

    if let Some(positions) = positions {
        let positions_values: Result<Vec<i32>, InferenceError> = (0..sequence_length)
            .map(|index| i32::try_from(index).map_err(|_| InferenceError::MemorySizeOverflow))
            .collect();
        positions
            .write_data_backend(&positions_values?)
            .map_err(|source| {
                InferenceError::ggml("Tensor::write_data_backend<POSITIONS>", source)
            })?;
    }
    if let Some(mask) = mask {
        let mask_values = build_causal_mask_values(
            sequence_length,
            sequence_length,
            match config.mask {
                AttentionMaskPolicy::None => 0,
                AttentionMaskPolicy::Causal { past_tokens } => past_tokens,
            },
        );
        mask.write_data_backend(&mask_values).map_err(|source| {
            InferenceError::ggml("Tensor::write_data_backend<CAUSAL_MASK>", source)
        })?;
    }

    for _ in 0..repeats {
        backend
            .compute(&mut graph)
            .map_err(|source| InferenceError::ggml("Backend::compute", source))?;
    }

    let output = graph
        .last_node()
        .map_err(|source| InferenceError::ggml("Graph::last_node", source))?
        .read_data_backend::<f32>()
        .map_err(|source| InferenceError::ggml("Tensor::read_data_backend", source))?;

    Ok(AttentionInferenceReport {
        backend_name,
        hidden_features,
        sequence_length,
        repeats,
        output,
    })
}

/// Builds reusable projected KV tensors for decode-like proxy runs.
pub fn build_attention_decode_cache(
    weights: &AttentionWeights,
    key_value_input: &[f32],
    key_value_length: usize,
) -> Result<AttentionDecodeCache, InferenceError> {
    StandardDecodeCacheBuilder::new(F32MatmulProjector).build_cache(
        weights,
        key_value_input,
        key_value_length,
    )
}

pub(crate) fn attention_decode_proxy_with_cache_repeats_inner(
    weights: &AttentionWeights,
    query_input: &[f32],
    cache: &AttentionDecodeCache,
    backend_kind: LlamaBackend,
    repeats: usize,
    causal_past_tokens: usize,
) -> Result<AttentionDecodeProxyReport, InferenceError> {
    let config = weights.config;
    let hidden_features = config.hidden_features();
    let query_length = config.sequence_length();
    let key_value_length = cache.key_value_length();
    let cache_kv_features = cache.kv_features();
    if cache_kv_features != config.kv_features() {
        return Err(InferenceError::InvalidInputLength {
            expected: config.kv_features(),
            actual: cache_kv_features,
        });
    }
    let expected_query_len = hidden_features
        .checked_mul(query_length)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    if query_input.len() != expected_query_len {
        return Err(InferenceError::InvalidInputLength {
            expected: expected_query_len,
            actual: query_input.len(),
        });
    }
    if repeats == 0 {
        return Err(InferenceError::InvalidRepeats);
    }

    let kv_features = config.kv_features();
    let expected_projected_len = kv_features
        .checked_mul(key_value_length)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    if cache.projected_k_values.len() != expected_projected_len {
        return Err(InferenceError::InvalidInputLength {
            expected: expected_projected_len,
            actual: cache.projected_k_values.len(),
        });
    }
    if cache.projected_v_values.len() != expected_projected_len {
        return Err(InferenceError::InvalidInputLength {
            expected: expected_projected_len,
            actual: cache.projected_v_values.len(),
        });
    }

    let ctx_size = recommended_attention_backend_memory_bytes_for_lengths(
        config,
        query_length,
        key_value_length,
    )?;
    let runtime = DefaultBackendRuntimeBuilder.build_runtime(backend_kind, ctx_size)?;
    let backend = runtime.backend;
    let backend_name = runtime.backend_name;
    let ctx = runtime.ctx;

    let query_features = config.query_features();

    let w_q = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, query_features))
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_Q>", source))?;
    let w_o = ctx
        .new_tensor_2d::<f32>(Shape2D::new(query_features, hidden_features))
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_O>", source))?;
    let x_q = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, query_length))
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<X_Q>", source))?;
    let k = ctx
        .new_tensor_2d::<f32>(Shape2D::new(kv_features, key_value_length))
        .map_err(|source| {
            InferenceError::ggml("Context::new_f32_tensor_2d_shape<K_CACHE>", source)
        })?;
    let v = ctx
        .new_tensor_2d::<f32>(Shape2D::new(kv_features, key_value_length))
        .map_err(|source| {
            InferenceError::ggml("Context::new_f32_tensor_2d_shape<V_CACHE>", source)
        })?;

    let q = ctx
        .mul_mat(&w_q, &x_q)
        .map_err(|source| InferenceError::ggml("Context::mul_mat(Q)", source))?;

    let (positions_q, positions_k) = if matches!(config.rotary, RotaryEmbedding::Llama(_)) {
        let positions_q = ctx
            .new_tensor_1d::<i32>(Length::new(query_length))
            .map_err(|source| {
                InferenceError::ggml("Context::new_i32_tensor_1d_len<QUERY_POS>", source)
            })?;
        let positions_k = ctx
            .new_tensor_1d::<i32>(Length::new(key_value_length))
            .map_err(|source| {
                InferenceError::ggml("Context::new_i32_tensor_1d_len<KV_POS>", source)
            })?;
        (Some(positions_q), Some(positions_k))
    } else {
        (None, None)
    };
    let mask = if matches!(config.mask, AttentionMaskPolicy::Causal { .. }) {
        Some(
            ctx.new_tensor_2d::<f32>(Shape2D::new(key_value_length, query_length))
                .map_err(|source| {
                    InferenceError::ggml("Context::new_f32_tensor_2d_shape<CAUSAL_MASK>", source)
                })?,
        )
    } else {
        None
    };

    let mut output_projection = None;
    let bytes_per_element = std::mem::size_of::<f32>();
    let q_row_stride = query_features
        .checked_mul(bytes_per_element)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    let kv_row_stride = kv_features
        .checked_mul(bytes_per_element)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    let o_row_stride = q_row_stride;
    let attention_scale = 1.0 / (config.head_dimension() as f32).sqrt();
    let rotary_applier = LlamaRotaryApplier;

    for head in 0..config.query_head_count() {
        let query_offset = head
            .checked_mul(config.head_dimension())
            .and_then(|value| value.checked_mul(bytes_per_element))
            .ok_or(InferenceError::MemorySizeOverflow)?;
        let kv_head = head / config.layout.kv_group_size();
        let kv_offset = kv_head
            .checked_mul(config.head_dimension())
            .and_then(|value| value.checked_mul(bytes_per_element))
            .ok_or(InferenceError::MemorySizeOverflow)?;

        let q_head = ctx
            .view_2d(
                &q,
                config.head_dimension(),
                query_length,
                q_row_stride,
                query_offset,
            )
            .map_err(|source| InferenceError::ggml("Context::view_2d(Q_HEAD)", source))?;
        let k_head = ctx
            .view_2d(
                &k,
                config.head_dimension(),
                key_value_length,
                kv_row_stride,
                kv_offset,
            )
            .map_err(|source| InferenceError::ggml("Context::view_2d(K_HEAD)", source))?;
        let v_head = ctx
            .view_2d(
                &v,
                config.head_dimension(),
                key_value_length,
                kv_row_stride,
                kv_offset,
            )
            .map_err(|source| InferenceError::ggml("Context::view_2d(V_HEAD)", source))?;

        let q_head = rotary_applier.apply_single_with_sequence(
            &ctx,
            &q_head,
            positions_q.as_ref(),
            config,
            query_length,
        )?;
        let k_head = rotary_applier.apply_single_with_sequence(
            &ctx,
            &k_head,
            positions_k.as_ref(),
            config,
            key_value_length,
        )?;

        let scores = ctx
            .mul_mat(&k_head, &q_head)
            .map_err(|source| InferenceError::ggml("Context::mul_mat(K_HEAD*Q_HEAD)", source))?;
        let probabilities = ctx
            .soft_max_ext(&scores, mask.as_ref(), attention_scale, 0.0)
            .map_err(|source| InferenceError::ggml("Context::soft_max_ext", source))?;
        let v_t = ctx
            .transpose(&v_head)
            .map_err(|source| InferenceError::ggml("Context::transpose(V_HEAD)", source))?;
        let v_t = ctx
            .cont(&v_t)
            .map_err(|source| InferenceError::ggml("Context::cont(V_HEAD)", source))?;
        let head_output = ctx
            .mul_mat(&v_t, &probabilities)
            .map_err(|source| InferenceError::ggml("Context::mul_mat(VT*P)", source))?;
        let w_o_head = ctx
            .view_2d(
                &w_o,
                config.head_dimension(),
                hidden_features,
                o_row_stride,
                query_offset,
            )
            .map_err(|source| InferenceError::ggml("Context::view_2d(W_O_HEAD)", source))?;
        let projected = ctx
            .mul_mat(&w_o_head, &head_output)
            .map_err(|source| InferenceError::ggml("Context::mul_mat(W_O_HEAD*HEAD)", source))?;

        output_projection = Some(if let Some(acc) = output_projection {
            ctx.add(&acc, &projected)
                .map_err(|source| InferenceError::ggml("Context::add(head_acc)", source))?
        } else {
            projected
        });
    }

    let y = output_projection.ok_or(InferenceError::InvalidAttentionLayout {
        hidden_features,
        query_head_count: config.query_head_count(),
        kv_head_count: config.kv_head_count(),
    })?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| InferenceError::ggml("Context::new_graph", source))?;
    graph.build_forward_expand(&y);
    let _buffer = ctx
        .allocate_tensors(&backend)
        .map_err(|source| InferenceError::ggml("Context::allocate_tensors", source))?;

    w_q.write_data_backend(weights.q_values())
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<W_Q>", source))?;
    w_o.write_data_backend(weights.o_values())
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<W_O>", source))?;
    x_q.write_data_backend(query_input)
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<X_Q>", source))?;
    k.write_data_backend(&cache.projected_k_values)
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<K_CACHE>", source))?;
    v.write_data_backend(&cache.projected_v_values)
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<V_CACHE>", source))?;

    if let Some(positions_q) = positions_q {
        let positions_values: Result<Vec<i32>, InferenceError> = (0..query_length)
            .map(|index| {
                let position = causal_past_tokens
                    .checked_add(index)
                    .ok_or(InferenceError::MemorySizeOverflow)?;
                i32::try_from(position).map_err(|_| InferenceError::MemorySizeOverflow)
            })
            .collect();
        positions_q
            .write_data_backend(&positions_values?)
            .map_err(|source| {
                InferenceError::ggml("Tensor::write_data_backend<QUERY_POS>", source)
            })?;
    }
    if let Some(positions_k) = positions_k {
        let positions_values: Result<Vec<i32>, InferenceError> = (0..key_value_length)
            .map(|index| i32::try_from(index).map_err(|_| InferenceError::MemorySizeOverflow))
            .collect();
        positions_k
            .write_data_backend(&positions_values?)
            .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<KV_POS>", source))?;
    }
    if let Some(mask) = mask {
        let mask_values =
            build_causal_mask_values(query_length, key_value_length, causal_past_tokens);
        mask.write_data_backend(&mask_values).map_err(|source| {
            InferenceError::ggml("Tensor::write_data_backend<CAUSAL_MASK>", source)
        })?;
    }

    for _ in 0..repeats {
        backend
            .compute(&mut graph)
            .map_err(|source| InferenceError::ggml("Backend::compute", source))?;
    }

    let output = graph
        .last_node()
        .map_err(|source| InferenceError::ggml("Graph::last_node", source))?
        .read_data_backend::<f32>()
        .map_err(|source| InferenceError::ggml("Tensor::read_data_backend", source))?;

    Ok(AttentionDecodeProxyReport {
        backend_name,
        hidden_features,
        query_length,
        key_value_length,
        repeats,
        output,
    })
}

pub(crate) fn build_causal_mask_values(
    query_length: usize,
    key_value_length: usize,
    past_tokens: usize,
) -> Vec<f32> {
    let mut values = vec![0.0_f32; query_length * key_value_length];
    fill_causal_mask_values(&mut values, query_length, key_value_length, past_tokens);
    values
}

pub(crate) fn fill_causal_mask_values(
    values: &mut [f32],
    query_length: usize,
    key_value_length: usize,
    past_tokens: usize,
) {
    debug_assert_eq!(values.len(), query_length * key_value_length);
    let mut offset = 0usize;
    for query in 0..query_length {
        let max_allowed_key = past_tokens.saturating_add(query);
        for key in 0..key_value_length {
            let allowed = key <= max_allowed_key;
            values[offset] = if allowed { 0.0 } else { -1.0e9 };
            offset += 1;
        }
    }
}

fn recommended_attention_backend_memory_bytes(
    config: AttentionInferenceConfig,
) -> Result<ggml_rs::Bytes, InferenceError> {
    let sequence_length = config.sequence_length();
    recommended_attention_backend_memory_bytes_for_lengths(config, sequence_length, sequence_length)
}

pub(crate) fn recommended_attention_backend_memory_bytes_for_lengths(
    config: AttentionInferenceConfig,
    query_length: usize,
    key_value_length: usize,
) -> Result<ggml_rs::Bytes, InferenceError> {
    let hidden_features = config.hidden_features();
    let query_features = config.query_features();
    let kv_features = config.kv_features();
    let head_dimension = config.head_dimension();

    let q_projection = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, query_features),
        Shape2D::new(hidden_features, query_length),
    )
    .map_err(|source| {
        InferenceError::ggml(
            "Context::recommended_backend_matmul_memory::<f32>(q_projection)",
            source,
        )
    })?;
    let kv_projection = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, kv_features),
        Shape2D::new(hidden_features, key_value_length),
    )
    .map_err(|source| {
        InferenceError::ggml(
            "Context::recommended_backend_matmul_memory::<f32>(kv_projection)",
            source,
        )
    })?;
    let score_matmul = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(head_dimension, key_value_length),
        Shape2D::new(head_dimension, query_length),
    )
    .map_err(|source| {
        InferenceError::ggml(
            "Context::recommended_backend_matmul_memory::<f32>(score)",
            source,
        )
    })?;
    let value_matmul = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(key_value_length, head_dimension),
        Shape2D::new(key_value_length, query_length),
    )
    .map_err(|source| {
        InferenceError::ggml(
            "Context::recommended_backend_matmul_memory::<f32>(value)",
            source,
        )
    })?;
    let output_projection = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(head_dimension, hidden_features),
        Shape2D::new(head_dimension, query_length),
    )
    .map_err(|source| {
        InferenceError::ggml(
            "Context::recommended_backend_matmul_memory::<f32>(output_projection)",
            source,
        )
    })?;

    let head_count = config.query_head_count();
    let head_contrib = score_matmul
        .get()
        .checked_add(value_matmul.get())
        .and_then(|value| value.checked_add(output_projection.get()))
        .ok_or(InferenceError::MemorySizeOverflow)?;
    let total = q_projection
        .get()
        .checked_add(kv_projection.get())
        .and_then(|value| value.checked_add(kv_projection.get()))
        .and_then(|value| {
            head_contrib
                .checked_mul(head_count)
                .and_then(|head| value.checked_add(head))
        })
        .and_then(|value| value.checked_add(4 * 1024 * 1024))
        .ok_or(InferenceError::MemorySizeOverflow)?;
    Ok(ggml_rs::Bytes::new(total))
}
