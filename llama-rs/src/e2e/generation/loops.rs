//! Core generation loops: full-reprocess and two-phase (prefill + decode).

use super::super::error::E2eError;
use super::super::numeric::checked_mul;
use super::super::state::GenerationState;
use super::super::tensor_ops::gather_embeddings;
use super::{
    DecodeStrategy, GenerationInputs, GenerationMode, GenerationOutput, InferenceStrategy,
    LmHeadResources, PersistentDecodeResources, PrefillStrategy, graph_sample_fallback,
    process_all_layers,
};

pub(in crate::e2e) fn generate_from_plans(
    inputs: &GenerationInputs<'_>,
    mode: GenerationMode,
) -> Result<GenerationOutput, E2eError> {
    let prompt_token_count = inputs.prompt_token_ids.len();
    if prompt_token_count == 0 {
        return Err(E2eError::EmptyPrompt);
    }

    let mut all_token_ids = vec![inputs.pad_token_id; inputs.total_sequence_length];
    all_token_ids[..prompt_token_count].copy_from_slice(inputs.prompt_token_ids);
    let mut generated_token_ids = Vec::with_capacity(inputs.max_new_tokens);
    let mut current_token_count = prompt_token_count;

    let effective_mode = match mode {
        GenerationMode::Auto => {
            if inputs.max_new_tokens == 0 {
                GenerationMode::FullReprocess
            } else {
                GenerationMode::TwoPhase
            }
        }
        other => other,
    };

    match effective_mode {
        GenerationMode::FullReprocess | GenerationMode::Auto => {
            full_reprocess_loop(
                inputs,
                &mut all_token_ids,
                &mut generated_token_ids,
                &mut current_token_count,
            )?;
        }
        GenerationMode::TwoPhase => {
            two_phase_loop(
                inputs,
                &mut all_token_ids,
                &mut generated_token_ids,
                &mut current_token_count,
            )?;
        }
    }

    Ok(GenerationOutput {
        generated_token_ids,
        all_token_ids: all_token_ids[..current_token_count].to_vec(),
    })
}

fn full_reprocess_loop(
    inputs: &GenerationInputs<'_>,
    all_token_ids: &mut [i32],
    generated_token_ids: &mut Vec<i32>,
    current_token_count: &mut usize,
) -> Result<(), E2eError> {
    let mut strategy = InferenceStrategy;

    // Persistent LM head: build graph and upload weights once.
    let mut lm_head = LmHeadResources::try_build(
        inputs.hidden_features,
        inputs.vocab_size,
        inputs.rms_norm_eps,
        inputs.output_weight_values,
        inputs.output_norm_values,
        inputs.backend,
    );

    for _step in 0..inputs.max_new_tokens {
        let active_token_ids = &all_token_ids[..*current_token_count];
        let mut hidden = gather_embeddings(
            inputs.token_embedding_values,
            inputs.hidden_features,
            inputs.vocab_size,
            active_token_ids,
        )?;

        process_all_layers(
            &mut hidden,
            inputs.layer_plans,
            &mut strategy,
            *current_token_count,
            inputs.rms_norm_eps,
            inputs.backend,
            &mut [],
        )?;

        let last_index = current_token_count
            .checked_sub(1)
            .ok_or(E2eError::EmptyPrompt)?;

        let next_token_id = if let Some(ref mut lm) = lm_head {
            let offset = checked_mul(last_index, inputs.hidden_features)?;
            let last_hidden = &hidden[offset..offset + inputs.hidden_features];
            lm.sample_hidden(last_hidden, inputs.backend)?
        } else {
            graph_sample_fallback(&hidden, last_index, inputs)?
        };

        generated_token_ids.push(next_token_id);
        if *current_token_count < inputs.total_sequence_length {
            all_token_ids[*current_token_count] = next_token_id;
            *current_token_count += 1;
        }

        if inputs.eos_token_id.is_some_and(|eos| eos == next_token_id) {
            break;
        }
    }
    Ok(())
}

fn two_phase_loop(
    inputs: &GenerationInputs<'_>,
    all_token_ids: &mut [i32],
    generated_token_ids: &mut Vec<i32>,
    current_token_count: &mut usize,
) -> Result<(), E2eError> {
    let prompt_token_count = inputs.prompt_token_ids.len();

    if inputs.max_new_tokens == 0 {
        return Ok(());
    }

    let mut state = GenerationState::new(inputs.layer_plans, inputs.total_sequence_length)?;

    // Build all persistent resources upfront (LM head, projections, KV caches,
    // scoring ctx, linear scratch, MLPs). LM head is reused for both prefill
    // sampling and decode loop — no duplicate graph build.
    let mut resources = LmHeadResources::try_build(
        inputs.hidden_features,
        inputs.vocab_size,
        inputs.rms_norm_eps,
        inputs.output_weight_values,
        inputs.output_norm_values,
        inputs.backend,
    )
    .map(|lm_head| {
        PersistentDecodeResources::try_build(
            inputs.layer_plans,
            lm_head,
            inputs.rms_norm_eps,
            inputs.total_sequence_length,
            inputs.backend,
        )
    });

    // Phase 1: Prefill — process all prompt tokens at once, capturing state.
    let prompt_ids = &all_token_ids[..prompt_token_count];
    let mut hidden = gather_embeddings(
        inputs.token_embedding_values,
        inputs.hidden_features,
        inputs.vocab_size,
        prompt_ids,
    )?;

    {
        let mut strategy = PrefillStrategy { state: &mut state };
        process_all_layers(
            &mut hidden,
            inputs.layer_plans,
            &mut strategy,
            prompt_token_count,
            inputs.rms_norm_eps,
            inputs.backend,
            &mut [],
        )?;
    }

    let last_index = prompt_token_count
        .checked_sub(1)
        .ok_or(E2eError::EmptyPrompt)?;

    // Sample first token using persistent LM head if available.
    let first_token_id = if let Some(ref mut res) = resources {
        res.sample_token(&hidden, last_index, inputs.hidden_features, inputs.backend)?
    } else {
        graph_sample_fallback(&hidden, last_index, inputs)?
    };

    generated_token_ids.push(first_token_id);
    all_token_ids[prompt_token_count] = first_token_id;
    *current_token_count = prompt_token_count + 1;

    if inputs.eos_token_id.is_some_and(|eos| eos == first_token_id) {
        return Ok(());
    }

    if inputs.max_new_tokens <= 1 {
        return Ok(());
    }

    // Seed persistent KV caches from host prefill state.
    if let Some(ref res) = resources {
        res.seed_kv_caches(&state);
    }

    // Phase 2: Decode — one token at a time using cached state.
    for _step in 1..inputs.max_new_tokens {
        let new_token_id = all_token_ids[*current_token_count - 1];
        let mut hidden = gather_embeddings(
            inputs.token_embedding_values,
            inputs.hidden_features,
            inputs.vocab_size,
            &[new_token_id],
        )?;

        let next_token_id = if let Some(ref mut res) = resources {
            res.decode_step(
                &mut hidden,
                inputs.layer_plans,
                &mut state,
                inputs.hidden_features,
                inputs.rms_norm_eps,
                inputs.backend,
            )?;
            res.sample_token(&hidden, 0, inputs.hidden_features, inputs.backend)?
        } else {
            let mut strategy = DecodeStrategy { state: &mut state };
            process_all_layers(
                &mut hidden,
                inputs.layer_plans,
                &mut strategy,
                1,
                inputs.rms_norm_eps,
                inputs.backend,
                &mut [],
            )?;
            graph_sample_fallback(&hidden, 0, inputs)?
        };

        generated_token_ids.push(next_token_id);
        if *current_token_count < inputs.total_sequence_length {
            all_token_ids[*current_token_count] = next_token_id;
            *current_token_count += 1;
        }

        if inputs.eos_token_id.is_some_and(|eos| eos == next_token_id) {
            break;
        }
    }
    Ok(())
}
