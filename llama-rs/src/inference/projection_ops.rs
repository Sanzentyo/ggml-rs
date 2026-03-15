use super::{AttentionDecodeCache, AttentionWeights, InferenceError};

pub(super) trait TensorProjector {
    fn project(
        &self,
        weights: &[f32],
        input: &[f32],
        hidden_features: usize,
        output_features: usize,
        sequence_length: usize,
    ) -> Result<Vec<f32>, InferenceError>;
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct F32MatmulProjector;

impl TensorProjector for F32MatmulProjector {
    fn project(
        &self,
        weights: &[f32],
        input: &[f32],
        hidden_features: usize,
        output_features: usize,
        sequence_length: usize,
    ) -> Result<Vec<f32>, InferenceError> {
        let expected_weights_len = hidden_features
            .checked_mul(output_features)
            .ok_or(InferenceError::MemorySizeOverflow)?;
        if weights.len() != expected_weights_len {
            return Err(InferenceError::InvalidInputLength {
                expected: expected_weights_len,
                actual: weights.len(),
            });
        }
        let expected_input_len = hidden_features
            .checked_mul(sequence_length)
            .ok_or(InferenceError::MemorySizeOverflow)?;
        if input.len() != expected_input_len {
            return Err(InferenceError::InvalidInputLength {
                expected: expected_input_len,
                actual: input.len(),
            });
        }
        let output_len = output_features
            .checked_mul(sequence_length)
            .ok_or(InferenceError::MemorySizeOverflow)?;
        let mut output = vec![0.0_f32; output_len];
        for seq in 0..sequence_length {
            let input_base = seq
                .checked_mul(hidden_features)
                .ok_or(InferenceError::MemorySizeOverflow)?;
            let out_base = seq
                .checked_mul(output_features)
                .ok_or(InferenceError::MemorySizeOverflow)?;
            for out_feature in 0..output_features {
                let weight_base = out_feature
                    .checked_mul(hidden_features)
                    .ok_or(InferenceError::MemorySizeOverflow)?;
                let mut acc = 0.0_f32;
                for hidden in 0..hidden_features {
                    acc += weights[weight_base + hidden] * input[input_base + hidden];
                }
                output[out_base + out_feature] = acc;
            }
        }
        Ok(output)
    }
}

pub(super) trait DecodeCacheBuilder {
    fn build_cache(
        &self,
        weights: &AttentionWeights,
        key_value_input: &[f32],
        key_value_length: usize,
    ) -> Result<AttentionDecodeCache, InferenceError>;
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct StandardDecodeCacheBuilder<P> {
    projector: P,
}

impl<P> StandardDecodeCacheBuilder<P> {
    pub(super) const fn new(projector: P) -> Self {
        Self { projector }
    }
}

impl<P> DecodeCacheBuilder for StandardDecodeCacheBuilder<P>
where
    P: TensorProjector,
{
    fn build_cache(
        &self,
        weights: &AttentionWeights,
        key_value_input: &[f32],
        key_value_length: usize,
    ) -> Result<AttentionDecodeCache, InferenceError> {
        let config = weights.config;
        let hidden_features = config.hidden_features();
        if key_value_length == 0 {
            return Err(InferenceError::InvalidAttentionShape {
                hidden_features,
                sequence_length: key_value_length,
            });
        }
        let expected_kv_input_len = hidden_features
            .checked_mul(key_value_length)
            .ok_or(InferenceError::MemorySizeOverflow)?;
        if key_value_input.len() != expected_kv_input_len {
            return Err(InferenceError::InvalidInputLength {
                expected: expected_kv_input_len,
                actual: key_value_input.len(),
            });
        }
        let kv_features = config.kv_features();
        let projected_k_values = self.projector.project(
            weights.k_values(),
            key_value_input,
            hidden_features,
            kv_features,
            key_value_length,
        )?;
        let projected_v_values = self.projector.project(
            weights.v_values(),
            key_value_input,
            hidden_features,
            kv_features,
            key_value_length,
        )?;
        Ok(AttentionDecodeCache {
            key_value_length,
            kv_features,
            projected_k_values,
            projected_v_values,
        })
    }
}
