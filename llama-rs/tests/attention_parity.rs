#![cfg(feature = "link-system")]

use llama_rs::{
    AttentionInferenceConfig, AttentionLayout, AttentionMaskPolicy, AttentionWeights, LlamaBackend,
    run_attention_inference_with_weights_repeats,
};
use std::error::Error;

#[test]
fn attention_cpu_matches_metal() -> Result<(), Box<dyn Error>> {
    let layout = AttentionLayout::from_hidden_features(8, 4, 2)?;
    let config = AttentionInferenceConfig::from_layout(layout, 4)?;
    let weights = AttentionWeights::deterministic(config);
    let input: Vec<f32> = (0..(config.hidden_features() * config.sequence_length()))
        .map(|index| ((index + 3) % 29) as f32 * 0.0625)
        .collect();

    let cpu = run_attention_inference_with_weights_repeats(&weights, &input, LlamaBackend::Cpu, 1)?;
    let metal = match run_attention_inference_with_weights_repeats(
        &weights,
        &input,
        LlamaBackend::Metal,
        1,
    ) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("metal backend unavailable; skipping attention parity: {error}");
            return Ok(());
        }
    };

    assert_eq!(cpu.output.len(), metal.output.len());
    let max_delta = cpu
        .output
        .iter()
        .zip(metal.output.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_delta <= 2e-3,
        "attention cpu/metal divergence too high: max_delta={max_delta}"
    );

    Ok(())
}

#[test]
fn attention_causal_cpu_runs() -> Result<(), Box<dyn Error>> {
    let layout = AttentionLayout::from_hidden_features(8, 4, 2)?;
    let config = AttentionInferenceConfig::from_layout(layout, 4)?
        .with_mask(AttentionMaskPolicy::Causal { past_tokens: 0 });
    let weights = AttentionWeights::deterministic(config);
    let input: Vec<f32> = (0..(config.hidden_features() * config.sequence_length()))
        .map(|index| ((index + 3) % 29) as f32 * 0.0625)
        .collect();

    let report =
        run_attention_inference_with_weights_repeats(&weights, &input, LlamaBackend::Cpu, 1)?;
    assert_eq!(report.output.len(), input.len());

    Ok(())
}
