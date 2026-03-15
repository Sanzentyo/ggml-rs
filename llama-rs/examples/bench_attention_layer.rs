//! Benchmark runner for minimal attention block inference.

use clap::Parser;
use llama_rs::{
    AttentionDecodeStepwiseConfig, AttentionHeadDimension, AttentionInferenceConfig,
    AttentionLayout, AttentionMaskPolicy, AttentionWeights, GgufModel, LlamaBackend,
    LlamaLayerTensorNames, MlpInferenceConfig, MlpWeights, RopeConfig, RotaryEmbedding,
    build_attention_decode_cache, resolve_mlp_weights_for_layer_auto, resolve_transformer_metadata,
    run_attention_decode_proxy_with_cache_repeats,
    run_attention_decode_stepwise_with_cache_repeats_with_block_mlp,
    run_attention_inference_with_weights_repeats,
};
use std::error::Error as StdError;
use std::str::FromStr;
use std::time::Instant;

fn build_stepwise_config(
    parsed: &ParsedArgs,
    backend: LlamaBackend,
    key_value_start: usize,
    steps: usize,
    repeats_per_step: usize,
    layer_repeat: usize,
) -> AttentionDecodeStepwiseConfig {
    AttentionDecodeStepwiseConfig::new(
        key_value_start,
        steps,
        parsed.causal_past_tokens,
        repeats_per_step,
    )
    .with_layer_repeat(layer_repeat)
    .with_kv_projection(parsed.decode_stepwise_kv_projection)
    .with_kv_cache_write(parsed.decode_stepwise_kv_cache_write)
    .with_kv_cache_write_to_cache(parsed.decode_stepwise_kv_cache_write_to_cache)
    .with_block_scope(parsed.decode_stepwise_block_scope)
    .with_sync_per_step(parsed.decode_stepwise_sync_step && matches!(backend, LlamaBackend::Metal))
    .with_readback_per_step(
        parsed.decode_stepwise_readback_step && matches!(backend, LlamaBackend::Metal),
    )
    .with_position_deltas(parsed.decode_stepwise_position_deltas)
    .with_mask_deltas(parsed.decode_stepwise_mask_deltas)
    .with_mask_host_buffer_elision(parsed.decode_stepwise_mask_host_buffer_elision)
    .with_fused_output_projection(parsed.decode_stepwise_fuse_output_projection)
    .with_static_kv_head_view_precompute(parsed.decode_stepwise_static_kv_head_precompute)
    .with_balanced_head_concat(parsed.decode_stepwise_balanced_head_concat)
    .with_head_output_staging_buffer(parsed.decode_stepwise_head_output_staging_buffer)
    .with_fused_block_gate_up_projection(parsed.decode_stepwise_fuse_block_gate_up)
}

fn stepwise_profile_label(parsed: &ParsedArgs, layer_repeat: usize) -> &'static str {
    if parsed.decode_stepwise_fuse_output_projection
        && parsed.decode_stepwise_layer_repeat_cpu == Some(5)
        && parsed.decode_stepwise_layer_repeat_metal == Some(6)
    {
        return "outproj_fused_balanced_cpu5_mtl6";
    }
    if parsed.decode_stepwise_fuse_output_projection
        && parsed.decode_stepwise_layer_repeat_cpu.is_none()
        && parsed.decode_stepwise_layer_repeat_metal.is_none()
        && layer_repeat == 5
    {
        "outproj_fused_layerx5"
    } else {
        "custom"
    }
}

fn main() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();
    let parsed = parse_args()?;
    let block_mlp_model = if let Some(model_path) = parsed.block_mlp_model_path.as_deref() {
        Some(GgufModel::open(model_path)?)
    } else {
        None
    };
    let model_layer_repeat = if parsed.decode_stepwise_layer_repeat_model {
        let model = block_mlp_model
            .as_ref()
            .ok_or("--decode-stepwise-layer-repeat-model requires --block-mlp-model")?;
        Some(resolve_transformer_metadata(model)?.block_count())
    } else {
        None
    };
    let block_layers: Vec<usize> = if let Some((start, end)) = parsed.block_mlp_layer_range {
        (start..=end).collect()
    } else {
        vec![parsed.block_mlp_layer]
    };
    let mut preflight_cpu_done = false;
    let mut preflight_metal_done = false;

    for case in &parsed.cases {
        if parsed.decode_steps.is_some() && case.sequence_length != 1 {
            return Err(format!(
                "--decode-steps requires q_seq=1 cases; got sequence length {}",
                case.sequence_length
            )
            .into());
        }
        let config = case.build_config(parsed.causal, parsed.rope, parsed.causal_past_tokens)?;
        let weights = AttentionWeights::deterministic(config);
        let query_input = case.build_input_with_length(case.sequence_length);
        let key_value_length = parsed.decode_kv_length.unwrap_or(case.sequence_length);
        let decode_cache_length = match parsed.decode_steps {
            Some(step_count) => key_value_length
                .checked_add(step_count - 1)
                .ok_or("decode cache length overflow")?,
            None => key_value_length,
        };
        let key_value_input = case.build_input_with_length(decode_cache_length);
        let decode_cache = parsed
            .decode_kv_length
            .map(|_| build_attention_decode_cache(&weights, &key_value_input, decode_cache_length))
            .transpose()?;
        for block_layer in block_layers.iter().copied() {
            let block_mlp_weights = if let Some(model) = block_mlp_model.as_ref() {
                Some(resolve_block_mlp_weights_for_layer(
                    model,
                    block_layer,
                    case.hidden_features,
                )?)
            } else {
                None
            };

            for backend in parsed.backends.iter().copied() {
                let layer_repeat =
                    model_layer_repeat.unwrap_or_else(|| parsed.layer_repeat_for_backend(backend));
                let preflight_done = match backend {
                    LlamaBackend::Cpu => &mut preflight_cpu_done,
                    LlamaBackend::Metal => &mut preflight_metal_done,
                };
                if parsed.decode_kv_length.is_some() {
                    let cache = decode_cache
                        .as_ref()
                        .ok_or("decode cache must be initialized when --decode-kv is set")?;
                    if let Some(step_count) = parsed.decode_steps {
                        if !*preflight_done {
                            let _ =
                                run_attention_decode_stepwise_with_cache_repeats_with_block_mlp(
                                    &weights,
                                    &query_input,
                                    cache,
                                    backend,
                                    build_stepwise_config(
                                        &parsed,
                                        backend,
                                        key_value_length,
                                        step_count,
                                        1,
                                        layer_repeat,
                                    ),
                                    block_mlp_weights.as_ref().map(|entry| &entry.0),
                                )?;
                            *preflight_done = true;
                        }
                        if parsed.warmup_iters > 0 {
                            let _ =
                                run_attention_decode_stepwise_with_cache_repeats_with_block_mlp(
                                    &weights,
                                    &query_input,
                                    cache,
                                    backend,
                                    build_stepwise_config(
                                        &parsed,
                                        backend,
                                        key_value_length,
                                        step_count,
                                        parsed.warmup_iters,
                                        layer_repeat,
                                    ),
                                    block_mlp_weights.as_ref().map(|entry| &entry.0),
                                )?;
                        }

                        let start = Instant::now();
                        let report =
                            run_attention_decode_stepwise_with_cache_repeats_with_block_mlp(
                                &weights,
                                &query_input,
                                cache,
                                backend,
                                build_stepwise_config(
                                    &parsed,
                                    backend,
                                    key_value_length,
                                    step_count,
                                    parsed.bench_iters,
                                    layer_repeat,
                                ),
                                block_mlp_weights.as_ref().map(|entry| &entry.0),
                            )?;
                        let elapsed = start.elapsed();
                        let total_step_iters = parsed
                            .bench_iters
                            .checked_mul(step_count)
                            .ok_or("stepwise iteration count overflow")?;
                        let avg_ms = elapsed.as_secs_f64() * 1000.0 / total_step_iters as f64;
                        let checksum: f64 = report
                            .output
                            .iter()
                            .take(16)
                            .map(|value| f64::from(*value))
                            .sum();

                        println!(
                            "[{}] attn decode stepwise bench hidden={} heads={}/{} q_seq={} kv_start={} steps={} past_start={} cache_reuse=true stepwise=true kv_proj={} kv_write={} kv_write_cache={} block={} block_mlp_real={} block_layer={} sync_step={} readback_step={} position_delta={} mask_delta={} mask_host_elide={} outproj_fused={} kvhead_static_precompute={} head_concat_balanced={} head_stage_buf={} block_gateup_fused={} profile={} layer_repeat={} rope={} warmup={} iters={} avg_token={:.3} ms checksum={:.6}",
                            report.backend_name,
                            case.hidden_features,
                            case.query_head_count,
                            case.kv_head_count,
                            case.sequence_length,
                            key_value_length,
                            step_count,
                            parsed.causal_past_tokens,
                            parsed.decode_stepwise_kv_projection,
                            parsed.decode_stepwise_kv_cache_write,
                            parsed.decode_stepwise_kv_cache_write_to_cache,
                            parsed.decode_stepwise_block_scope,
                            block_mlp_weights.as_ref().is_some_and(|entry| entry.1),
                            block_layer,
                            parsed.decode_stepwise_sync_step,
                            parsed.decode_stepwise_readback_step,
                            parsed.decode_stepwise_position_deltas,
                            parsed.decode_stepwise_mask_deltas,
                            parsed.decode_stepwise_mask_host_buffer_elision,
                            parsed.decode_stepwise_fuse_output_projection,
                            parsed.decode_stepwise_static_kv_head_precompute,
                            parsed.decode_stepwise_balanced_head_concat,
                            parsed.decode_stepwise_head_output_staging_buffer,
                            parsed.decode_stepwise_fuse_block_gate_up,
                            stepwise_profile_label(&parsed, layer_repeat),
                            layer_repeat,
                            parsed.rope,
                            parsed.warmup_iters,
                            parsed.bench_iters,
                            avg_ms,
                            checksum
                        );
                    } else {
                        if !*preflight_done {
                            run_attention_decode_proxy_with_cache_repeats(
                                &weights,
                                &query_input,
                                cache,
                                backend,
                                1,
                            )?;
                            *preflight_done = true;
                        }
                        if parsed.warmup_iters > 0 {
                            run_attention_decode_proxy_with_cache_repeats(
                                &weights,
                                &query_input,
                                cache,
                                backend,
                                parsed.warmup_iters,
                            )?;
                        }
                        let start = Instant::now();
                        let report = run_attention_decode_proxy_with_cache_repeats(
                            &weights,
                            &query_input,
                            cache,
                            backend,
                            parsed.bench_iters,
                        )?;
                        let elapsed = start.elapsed();
                        let avg_ms = elapsed.as_secs_f64() * 1000.0 / parsed.bench_iters as f64;
                        let checksum: f64 = report
                            .output
                            .iter()
                            .take(16)
                            .map(|value| f64::from(*value))
                            .sum();

                        println!(
                            "[{}] attn decode bench hidden={} heads={}/{} q_seq={} kv_seq={} past={} cache_reuse=true rope={} warmup={} iters={} avg={:.3} ms checksum={:.6}",
                            report.backend_name,
                            case.hidden_features,
                            case.query_head_count,
                            case.kv_head_count,
                            report.query_length,
                            report.key_value_length,
                            parsed.causal_past_tokens,
                            parsed.rope,
                            parsed.warmup_iters,
                            parsed.bench_iters,
                            avg_ms,
                            checksum
                        );
                    }
                } else {
                    if !*preflight_done {
                        run_attention_inference_with_weights_repeats(
                            &weights,
                            &query_input,
                            backend,
                            1,
                        )?;
                        *preflight_done = true;
                    }
                    if parsed.warmup_iters > 0 {
                        run_attention_inference_with_weights_repeats(
                            &weights,
                            &query_input,
                            backend,
                            parsed.warmup_iters,
                        )?;
                    }

                    let start = Instant::now();
                    let report = run_attention_inference_with_weights_repeats(
                        &weights,
                        &query_input,
                        backend,
                        parsed.bench_iters,
                    )?;
                    let elapsed = start.elapsed();
                    let avg_ms = elapsed.as_secs_f64() * 1000.0 / parsed.bench_iters as f64;
                    let checksum: f64 = report
                        .output
                        .iter()
                        .take(16)
                        .map(|value| f64::from(*value))
                        .sum();

                    println!(
                        "[{}] attn bench hidden={} heads={}/{} seq={} causal={} rope={} warmup={} iters={} avg={:.3} ms checksum={:.6}",
                        report.backend_name,
                        case.hidden_features,
                        case.query_head_count,
                        case.kv_head_count,
                        case.sequence_length,
                        parsed.causal,
                        parsed.rope,
                        parsed.warmup_iters,
                        parsed.bench_iters,
                        avg_ms,
                        checksum
                    );
                }
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct AttentionBenchCase {
    hidden_features: usize,
    query_head_count: usize,
    kv_head_count: usize,
    sequence_length: usize,
}

impl AttentionBenchCase {
    fn build_config(
        self,
        causal: bool,
        rope: bool,
        causal_past_tokens: usize,
    ) -> Result<AttentionInferenceConfig, Box<dyn StdError>> {
        let layout = AttentionLayout::from_hidden_features(
            self.hidden_features,
            self.query_head_count,
            self.kv_head_count,
        )?;
        let mut config = AttentionInferenceConfig::from_layout(layout, self.sequence_length)?;
        if causal {
            config = config.with_mask(AttentionMaskPolicy::Causal {
                past_tokens: causal_past_tokens,
            });
        }
        if rope {
            let rope_dimensions = AttentionHeadDimension::new(layout.head_dimension())?;
            config = config.with_rotary(RotaryEmbedding::Llama(RopeConfig {
                dimensions: rope_dimensions,
                base: 10_000.0,
                scale: 1.0,
                original_context: None,
            }));
        }
        Ok(config)
    }

    fn build_input_with_length(self, sequence_length: usize) -> Vec<f32> {
        (0..(self.hidden_features * sequence_length))
            .map(|index| ((index + 3) % 29) as f32 * 0.0625)
            .collect()
    }
}

#[derive(Debug)]
struct ParsedArgs {
    cases: Vec<AttentionBenchCase>,
    warmup_iters: usize,
    bench_iters: usize,
    causal: bool,
    causal_past_tokens: usize,
    rope: bool,
    decode_kv_length: Option<usize>,
    decode_steps: Option<usize>,
    decode_stepwise_kv_projection: bool,
    decode_stepwise_kv_cache_write: bool,
    decode_stepwise_kv_cache_write_to_cache: bool,
    decode_stepwise_block_scope: bool,
    decode_stepwise_sync_step: bool,
    decode_stepwise_readback_step: bool,
    decode_stepwise_position_deltas: bool,
    decode_stepwise_mask_deltas: bool,
    decode_stepwise_mask_host_buffer_elision: bool,
    decode_stepwise_fuse_output_projection: bool,
    decode_stepwise_static_kv_head_precompute: bool,
    decode_stepwise_balanced_head_concat: bool,
    decode_stepwise_head_output_staging_buffer: bool,
    decode_stepwise_fuse_block_gate_up: bool,
    decode_stepwise_layer_repeat: usize,
    decode_stepwise_layer_repeat_cpu: Option<usize>,
    decode_stepwise_layer_repeat_metal: Option<usize>,
    decode_stepwise_layer_repeat_model: bool,
    block_mlp_model_path: Option<String>,
    block_mlp_layer: usize,
    block_mlp_layer_range: Option<(usize, usize)>,
    backends: Vec<LlamaBackend>,
}

impl ParsedArgs {
    fn layer_repeat_for_backend(&self, backend: LlamaBackend) -> usize {
        match backend {
            LlamaBackend::Cpu => self
                .decode_stepwise_layer_repeat_cpu
                .unwrap_or(self.decode_stepwise_layer_repeat),
            LlamaBackend::Metal => self
                .decode_stepwise_layer_repeat_metal
                .unwrap_or(self.decode_stepwise_layer_repeat),
        }
    }
}

fn parse_args() -> Result<ParsedArgs, Box<dyn StdError>> {
    let cli = Cli::parse();
    let mut decode_stepwise_position_deltas = true;
    if cli.decode_stepwise_no_position_delta {
        decode_stepwise_position_deltas = false;
    }
    if cli.decode_stepwise_position_delta {
        decode_stepwise_position_deltas = true;
    }

    let decode_stepwise_mask_deltas = !cli.decode_stepwise_no_mask_delta;

    let mut decode_stepwise_mask_host_buffer_elision = false;
    if cli.decode_stepwise_elide_mask_host_buffer {
        decode_stepwise_mask_host_buffer_elision = true;
    }
    if cli.decode_stepwise_keep_mask_host_buffer {
        decode_stepwise_mask_host_buffer_elision = false;
    }

    let mut decode_stepwise_fuse_output_projection = false;
    if cli.decode_stepwise_fuse_output_proj {
        decode_stepwise_fuse_output_projection = true;
    }
    if cli.decode_stepwise_no_fuse_output_proj {
        decode_stepwise_fuse_output_projection = false;
    }

    let mut decode_stepwise_static_kv_head_precompute = true;
    if cli.decode_stepwise_no_static_kv_head_precompute {
        decode_stepwise_static_kv_head_precompute = false;
    }
    if cli.decode_stepwise_static_kv_head_precompute {
        decode_stepwise_static_kv_head_precompute = true;
    }

    let mut decode_stepwise_balanced_head_concat = false;
    if cli.decode_stepwise_balanced_head_concat {
        decode_stepwise_balanced_head_concat = true;
    }
    if cli.decode_stepwise_no_balanced_head_concat {
        decode_stepwise_balanced_head_concat = false;
    }

    let mut decode_stepwise_head_output_staging_buffer = false;
    if cli.decode_stepwise_head_stage_buffer {
        decode_stepwise_head_output_staging_buffer = true;
    }
    if cli.decode_stepwise_no_head_stage_buffer {
        decode_stepwise_head_output_staging_buffer = false;
    }

    let mut decode_stepwise_fuse_block_gate_up = false;
    if cli.decode_stepwise_fuse_block_gate_up {
        decode_stepwise_fuse_block_gate_up = true;
    }
    if cli.decode_stepwise_no_fuse_block_gate_up {
        decode_stepwise_fuse_block_gate_up = false;
    }

    let mut decode_stepwise_layer_repeat = cli.decode_stepwise_layer_repeat;
    let mut decode_stepwise_layer_repeat_cpu = cli.decode_stepwise_layer_repeat_cpu;
    let mut decode_stepwise_layer_repeat_metal = cli.decode_stepwise_layer_repeat_metal;
    if cli.decode_stepwise_profile_outproj_fused_layerx5 {
        decode_stepwise_fuse_output_projection = true;
        decode_stepwise_layer_repeat = 5;
        decode_stepwise_layer_repeat_cpu = None;
        decode_stepwise_layer_repeat_metal = None;
        decode_stepwise_static_kv_head_precompute = true;
        decode_stepwise_position_deltas = true;
        decode_stepwise_balanced_head_concat = false;
        decode_stepwise_head_output_staging_buffer = false;
        decode_stepwise_fuse_block_gate_up = false;
    }
    if cli.decode_stepwise_profile_outproj_fused_balanced {
        decode_stepwise_fuse_output_projection = true;
        decode_stepwise_layer_repeat = 5;
        decode_stepwise_layer_repeat_cpu = Some(5);
        decode_stepwise_layer_repeat_metal = Some(6);
        decode_stepwise_static_kv_head_precompute = true;
        decode_stepwise_position_deltas = true;
        decode_stepwise_balanced_head_concat = false;
        decode_stepwise_head_output_staging_buffer = false;
        decode_stepwise_fuse_block_gate_up = false;
    }

    let block_mlp_layer_range = cli
        .block_mlp_layer_range
        .as_deref()
        .map(|value| parse_layer_range_arg("--block-mlp-layer-range", value))
        .transpose()?;

    let cases = if let Some(value) = cli.cases.as_deref() {
        parse_cases_arg(value)?
    } else {
        vec![AttentionBenchCase {
            hidden_features: cli.hidden_features,
            query_head_count: cli.query_head_count,
            kv_head_count: cli.kv_head_count,
            sequence_length: cli.sequence_length,
        }]
    };

    let backends = if cli.backends.is_empty() {
        vec![LlamaBackend::Cpu, LlamaBackend::Metal]
    } else {
        cli.backends.into_iter().map(|backend| backend.0).collect()
    };

    if cli.bench_iters == 0 {
        return Err("--iters must be greater than zero".into());
    }
    if matches!(cli.decode_kv_length, Some(0)) {
        return Err("--decode-kv must be greater than zero".into());
    }
    if matches!(cli.decode_steps, Some(0)) {
        return Err("--decode-steps must be greater than zero".into());
    }
    if decode_stepwise_layer_repeat == 0 {
        return Err("--decode-stepwise-layer-repeat must be greater than zero".into());
    }
    if matches!(decode_stepwise_layer_repeat_cpu, Some(0)) {
        return Err("--decode-stepwise-layer-repeat-cpu must be greater than zero".into());
    }
    if matches!(decode_stepwise_layer_repeat_metal, Some(0)) {
        return Err("--decode-stepwise-layer-repeat-metal must be greater than zero".into());
    }
    if cli.decode_steps.is_some() && cli.decode_kv_length.is_none() {
        return Err("--decode-steps requires --decode-kv".into());
    }
    if cli.decode_steps.is_some() && !cli.causal {
        return Err("--decode-steps requires --causal".into());
    }
    if cli.decode_stepwise_kv_cache_write && !cli.decode_stepwise_kv_projection {
        return Err("--decode-stepwise-kv-cache-write requires --decode-stepwise-kv-proj".into());
    }
    if cli.decode_stepwise_kv_cache_write_to_cache && !cli.decode_stepwise_kv_cache_write {
        return Err(
            "--decode-stepwise-kv-cache-write-to-cache requires --decode-stepwise-kv-cache-write"
                .into(),
        );
    }
    if cli.block_mlp_model_path.is_some() && cli.decode_steps.is_none() {
        return Err("--block-mlp-model requires --decode-steps".into());
    }
    if cli.block_mlp_model_path.is_some() && !cli.decode_stepwise_block_scope {
        return Err("--block-mlp-model requires --decode-stepwise-block".into());
    }
    if cli.decode_stepwise_layer_repeat_model && cli.block_mlp_model_path.is_none() {
        return Err("--decode-stepwise-layer-repeat-model requires --block-mlp-model".into());
    }
    if block_mlp_layer_range.is_some() && cli.block_mlp_model_path.is_none() {
        return Err("--block-mlp-layer-range requires --block-mlp-model".into());
    }

    Ok(ParsedArgs {
        cases,
        warmup_iters: cli.warmup_iters,
        bench_iters: cli.bench_iters,
        causal: cli.causal,
        causal_past_tokens: cli.causal_past_tokens,
        rope: cli.rope,
        decode_kv_length: cli.decode_kv_length,
        decode_steps: cli.decode_steps,
        decode_stepwise_kv_projection: cli.decode_stepwise_kv_projection,
        decode_stepwise_kv_cache_write: cli.decode_stepwise_kv_cache_write,
        decode_stepwise_kv_cache_write_to_cache: cli.decode_stepwise_kv_cache_write_to_cache,
        decode_stepwise_block_scope: cli.decode_stepwise_block_scope,
        decode_stepwise_sync_step: cli.decode_stepwise_sync_step,
        decode_stepwise_readback_step: cli.decode_stepwise_readback_step,
        decode_stepwise_position_deltas,
        decode_stepwise_mask_deltas,
        decode_stepwise_mask_host_buffer_elision,
        decode_stepwise_fuse_output_projection,
        decode_stepwise_static_kv_head_precompute,
        decode_stepwise_balanced_head_concat,
        decode_stepwise_head_output_staging_buffer,
        decode_stepwise_fuse_block_gate_up,
        decode_stepwise_layer_repeat,
        decode_stepwise_layer_repeat_cpu,
        decode_stepwise_layer_repeat_metal,
        decode_stepwise_layer_repeat_model: cli.decode_stepwise_layer_repeat_model,
        block_mlp_model_path: cli.block_mlp_model_path,
        block_mlp_layer: cli.block_mlp_layer,
        block_mlp_layer_range,
        backends,
    })
}

#[derive(Debug, Clone, Parser)]
#[command(about = "Benchmark attention layer paths", version)]
struct Cli {
    #[arg(long = "hidden", default_value_t = 64)]
    hidden_features: usize,
    #[arg(long = "q-heads", default_value_t = 8)]
    query_head_count: usize,
    #[arg(long = "kv-heads", default_value_t = 8)]
    kv_head_count: usize,
    #[arg(long = "seq", default_value_t = 8)]
    sequence_length: usize,
    #[arg(long = "cases")]
    cases: Option<String>,
    #[arg(long = "warmup", default_value_t = 3)]
    warmup_iters: usize,
    #[arg(long = "iters", default_value_t = 30)]
    bench_iters: usize,
    #[arg(long = "causal")]
    causal: bool,
    #[arg(long = "past", default_value_t = 0)]
    causal_past_tokens: usize,
    #[arg(long = "rope")]
    rope: bool,
    #[arg(long = "decode-kv")]
    decode_kv_length: Option<usize>,
    #[arg(long = "decode-steps")]
    decode_steps: Option<usize>,
    #[arg(long = "decode-stepwise-kv-proj")]
    decode_stepwise_kv_projection: bool,
    #[arg(long = "decode-stepwise-kv-cache-write")]
    decode_stepwise_kv_cache_write: bool,
    #[arg(long = "decode-stepwise-kv-cache-write-to-cache")]
    decode_stepwise_kv_cache_write_to_cache: bool,
    #[arg(long = "decode-stepwise-block")]
    decode_stepwise_block_scope: bool,
    #[arg(long = "decode-stepwise-sync-step")]
    decode_stepwise_sync_step: bool,
    #[arg(long = "decode-stepwise-readback-step")]
    decode_stepwise_readback_step: bool,
    #[arg(
        long = "decode-stepwise-position-delta",
        conflicts_with = "decode_stepwise_no_position_delta"
    )]
    decode_stepwise_position_delta: bool,
    #[arg(
        long = "decode-stepwise-no-position-delta",
        conflicts_with = "decode_stepwise_position_delta"
    )]
    decode_stepwise_no_position_delta: bool,
    #[arg(long = "decode-stepwise-no-mask-delta")]
    decode_stepwise_no_mask_delta: bool,
    #[arg(
        long = "decode-stepwise-elide-mask-host-buffer",
        conflicts_with = "decode_stepwise_keep_mask_host_buffer"
    )]
    decode_stepwise_elide_mask_host_buffer: bool,
    #[arg(
        long = "decode-stepwise-keep-mask-host-buffer",
        conflicts_with = "decode_stepwise_elide_mask_host_buffer"
    )]
    decode_stepwise_keep_mask_host_buffer: bool,
    #[arg(
        long = "decode-stepwise-fuse-output-proj",
        conflicts_with = "decode_stepwise_no_fuse_output_proj"
    )]
    decode_stepwise_fuse_output_proj: bool,
    #[arg(
        long = "decode-stepwise-no-fuse-output-proj",
        conflicts_with = "decode_stepwise_fuse_output_proj"
    )]
    decode_stepwise_no_fuse_output_proj: bool,
    #[arg(
        long = "decode-stepwise-static-kv-head-precompute",
        conflicts_with = "decode_stepwise_no_static_kv_head_precompute"
    )]
    decode_stepwise_static_kv_head_precompute: bool,
    #[arg(
        long = "decode-stepwise-no-static-kv-head-precompute",
        conflicts_with = "decode_stepwise_static_kv_head_precompute"
    )]
    decode_stepwise_no_static_kv_head_precompute: bool,
    #[arg(
        long = "decode-stepwise-balanced-head-concat",
        conflicts_with = "decode_stepwise_no_balanced_head_concat"
    )]
    decode_stepwise_balanced_head_concat: bool,
    #[arg(
        long = "decode-stepwise-no-balanced-head-concat",
        conflicts_with = "decode_stepwise_balanced_head_concat"
    )]
    decode_stepwise_no_balanced_head_concat: bool,
    #[arg(
        long = "decode-stepwise-head-stage-buffer",
        conflicts_with = "decode_stepwise_no_head_stage_buffer"
    )]
    decode_stepwise_head_stage_buffer: bool,
    #[arg(
        long = "decode-stepwise-no-head-stage-buffer",
        conflicts_with = "decode_stepwise_head_stage_buffer"
    )]
    decode_stepwise_no_head_stage_buffer: bool,
    #[arg(
        long = "decode-stepwise-fuse-block-gate-up",
        conflicts_with = "decode_stepwise_no_fuse_block_gate_up"
    )]
    decode_stepwise_fuse_block_gate_up: bool,
    #[arg(
        long = "decode-stepwise-no-fuse-block-gate-up",
        conflicts_with = "decode_stepwise_fuse_block_gate_up"
    )]
    decode_stepwise_no_fuse_block_gate_up: bool,
    #[arg(long = "decode-stepwise-layer-repeat", default_value_t = 1)]
    decode_stepwise_layer_repeat: usize,
    #[arg(long = "decode-stepwise-layer-repeat-cpu")]
    decode_stepwise_layer_repeat_cpu: Option<usize>,
    #[arg(long = "decode-stepwise-layer-repeat-metal")]
    decode_stepwise_layer_repeat_metal: Option<usize>,
    #[arg(long = "decode-stepwise-layer-repeat-model")]
    decode_stepwise_layer_repeat_model: bool,
    #[arg(
        long = "decode-stepwise-profile-outproj-fused-layerx5",
        conflicts_with = "decode_stepwise_profile_outproj_fused_balanced"
    )]
    decode_stepwise_profile_outproj_fused_layerx5: bool,
    #[arg(
        long = "decode-stepwise-profile-outproj-fused-balanced",
        conflicts_with = "decode_stepwise_profile_outproj_fused_layerx5"
    )]
    decode_stepwise_profile_outproj_fused_balanced: bool,
    #[arg(long = "block-mlp-model")]
    block_mlp_model_path: Option<String>,
    #[arg(long = "block-mlp-layer", default_value_t = 0)]
    block_mlp_layer: usize,
    #[arg(long = "block-mlp-layer-range")]
    block_mlp_layer_range: Option<String>,
    backends: Vec<BackendArg>,
}

#[derive(Debug, Clone, Copy)]
struct BackendArg(LlamaBackend);

impl FromStr for BackendArg {
    type Err = <LlamaBackend as FromStr>::Err;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        LlamaBackend::from_str(s).map(Self)
    }
}

fn resolve_block_mlp_weights_for_layer(
    model: &GgufModel,
    layer: usize,
    hidden_features: usize,
) -> Result<(MlpWeights, bool), Box<dyn StdError>> {
    if let Ok(weights) = resolve_mlp_weights_for_layer_auto(model, layer) {
        return Ok((weights, true));
    }

    let gate_candidates = vec![
        format!("blk.{layer}.ffn_gate.weight"),
        format!("layers.{layer}.feed_forward.w1.weight"),
        format!("model.layers.{layer}.mlp.gate_proj.weight"),
        format!("model.layers.{layer}.mlp.w1.weight"),
    ];
    let up_candidates = vec![
        format!("blk.{layer}.ffn_up.weight"),
        format!("layers.{layer}.feed_forward.w3.weight"),
        format!("model.layers.{layer}.mlp.up_proj.weight"),
        format!("model.layers.{layer}.mlp.w3.weight"),
    ];
    let down_candidates = vec![
        format!("blk.{layer}.ffn_down.weight"),
        format!("layers.{layer}.feed_forward.w2.weight"),
        format!("model.layers.{layer}.mlp.down_proj.weight"),
        format!("model.layers.{layer}.mlp.w2.weight"),
    ];

    let gate = first_existing_tensor(model, &gate_candidates).ok_or_else(|| {
        format!(
            "failed to resolve block MLP gate tensor for layer {layer}; tried: {}",
            gate_candidates.join(", ")
        )
    })?;
    let up = first_existing_tensor(model, &up_candidates).ok_or_else(|| {
        format!(
            "failed to resolve block MLP up tensor for layer {layer}; tried: {}",
            up_candidates.join(", ")
        )
    })?;
    let down = first_existing_tensor(model, &down_candidates).ok_or_else(|| {
        format!(
            "failed to resolve block MLP down tensor for layer {layer}; tried: {}",
            down_candidates.join(", ")
        )
    })?;

    let names = LlamaLayerTensorNames {
        layer,
        attn_norm: String::new(),
        attn_q: String::new(),
        attn_k: String::new(),
        attn_v: String::new(),
        attn_output: String::new(),
        ffn_norm: String::new(),
        ffn_gate: gate,
        ffn_up: up,
        ffn_down: down,
    };
    match MlpWeights::from_model_layer(model, &names, hidden_features) {
        Ok(weights) => Ok((weights, true)),
        Err(_) => {
            // Last-resort fallback: keep real model-derived MLP shape and use
            // deterministic values when model decoding still fails.
            let metadata = resolve_transformer_metadata(model).map_err(|error| {
                format!("failed to resolve metadata for block MLP fallback: {error}")
            })?;
            let ffn_features = metadata.feed_forward_length().ok_or_else(|| {
                "metadata does not expose feed_forward_length for block MLP fallback".to_string()
            })?;
            let config = MlpInferenceConfig::new(hidden_features, ffn_features)?;
            let mut weights = MlpWeights::deterministic(config);
            weights.gate_tensor_name = names.ffn_gate;
            weights.up_tensor_name = names.ffn_up;
            weights.down_tensor_name = names.ffn_down;
            Ok((weights, false))
        }
    }
}

fn first_existing_tensor(model: &GgufModel, candidates: &[String]) -> Option<String> {
    candidates
        .iter()
        .find(|name| model.tensor_info(name).is_ok())
        .cloned()
}

fn parse_usize_arg(flag: &str, value: &str) -> Result<usize, Box<dyn StdError>> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid value for {flag}: {value} ({error})").into())
}

fn parse_layer_range_arg(flag: &str, value: &str) -> Result<(usize, usize), Box<dyn StdError>> {
    let Some((start, end)) = value.split_once(':') else {
        return Err(format!("invalid value for {flag}: {value} (expected start:end)").into());
    };
    let start = parse_usize_arg(flag, start)?;
    let end = parse_usize_arg(flag, end)?;
    if start > end {
        return Err(format!("invalid value for {flag}: {value} (start must be <= end)").into());
    }
    Ok((start, end))
}

fn parse_cases_arg(value: &str) -> Result<Vec<AttentionBenchCase>, Box<dyn StdError>> {
    let mut cases = Vec::new();
    for token in value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        let mut parts = token.split('x');
        let hidden_features = parts
            .next()
            .ok_or_else(|| format!("invalid case `{token}` (expected HxQxKxS)"))?;
        let query_head_count = parts
            .next()
            .ok_or_else(|| format!("invalid case `{token}` (expected HxQxKxS)"))?;
        let kv_head_count = parts
            .next()
            .ok_or_else(|| format!("invalid case `{token}` (expected HxQxKxS)"))?;
        let sequence_length = parts
            .next()
            .ok_or_else(|| format!("invalid case `{token}` (expected HxQxKxS)"))?;
        if parts.next().is_some() {
            return Err(format!("invalid case `{token}` (expected HxQxKxS)").into());
        }
        cases.push(AttentionBenchCase {
            hidden_features: parse_usize_arg("--cases", hidden_features)?,
            query_head_count: parse_usize_arg("--cases", query_head_count)?,
            kv_head_count: parse_usize_arg("--cases", kv_head_count)?,
            sequence_length: parse_usize_arg("--cases", sequence_length)?,
        });
    }
    if cases.is_empty() {
        return Err("at least one case must be provided for --cases".into());
    }
    Ok(cases)
}
