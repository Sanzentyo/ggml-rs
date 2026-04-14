mod host_ops;
mod lm_head;
mod normalization;
mod persistent_decode;
mod projection;

// Re-exports: normalization
pub(super) use normalization::{
    per_head_l2_norm, per_head_rms_norm, rms_norm_single, rms_norm_single_into,
    rms_norm_with_weight,
};

// Re-exports: host operations
pub(super) use host_ops::{
    add_in_place, gather_embeddings, head_slice, head_slice_mut, project_sequence,
};

// Re-exports: projection utilities
pub(super) use projection::{
    MATMUL_GRAPH_SLACK_BYTES, ProjectionSpec, build_batch_projections, execute_batch_projections,
    project_sequence_graph, upload_weight,
};

// Re-exports: LM head
#[cfg(test)]
pub(super) use lm_head::{argmax_token_id, lm_head_graph};
pub(super) use lm_head::{build_lm_head_graph, lm_head_sample_step, recommended_lm_head_memory};

// Re-exports: persistent decode projections
pub(super) use persistent_decode::{
    PersistentDecodeProjection, build_persistent_full_attention_graphs,
    build_persistent_linear_attention_graphs, recommended_persistent_full_attention_memory,
    recommended_persistent_linear_attention_memory,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ggml_rs::{Backend, Context};

    #[test]
    fn rms_norm_applies_weight_per_position() {
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0_f32, 0.25];
        let output =
            rms_norm_with_weight(&input, 2, 2, &weight, 1e-5).expect("rms norm should succeed");
        assert_eq!(output.len(), input.len());
        assert!(output[0].is_finite());
        assert!(output[1].is_finite());
        assert!(output[2].is_finite());
        assert!(output[3].is_finite());
        assert!(output[0].abs() > output[1].abs());
    }

    #[test]
    fn rms_norm_eps_changes_scaled_output() {
        let input = vec![1.0_f32, 2.0];
        let weight = vec![1.0_f32, 1.0];
        let loose = rms_norm_with_weight(&input, 2, 1, &weight, 1e-5).expect("rms norm");
        let tight = rms_norm_with_weight(&input, 2, 1, &weight, 1e-6).expect("rms norm");
        assert_ne!(loose, tight);
    }

    #[test]
    fn project_sequence_graph_matches_host_projection() {
        use crate::backend::ensure_backends_loaded;
        use ggml_rs::BackendKind;

        let input_features = 8_usize;
        let output_features = 4_usize;
        let seq_len = 3_usize;

        let weight: Vec<f32> = (0..output_features * input_features)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();
        let input: Vec<f32> = (0..seq_len * input_features)
            .map(|i| (i as f32 + 1.0) * 0.05)
            .collect();

        let host_result =
            project_sequence(&input, seq_len, input_features, output_features, &weight)
                .expect("host projection");

        ensure_backends_loaded();
        let backend = Backend::new(BackendKind::Cpu).expect("CPU backend");

        let graph_result = project_sequence_graph(
            &input,
            seq_len,
            input_features,
            output_features,
            &weight,
            &backend,
        )
        .expect("graph projection");

        assert_eq!(host_result.len(), graph_result.len());
        for (i, (h, g)) in host_result.iter().zip(graph_result.iter()).enumerate() {
            assert!(
                (h - g).abs() < 1e-5,
                "element {i}: host={h} vs graph={g}, diff={}",
                (h - g).abs()
            );
        }
    }

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax_token_id(&[1.0, 3.0, 2.0]).unwrap(), 1);
        assert_eq!(argmax_token_id(&[5.0, 1.0, 2.0]).unwrap(), 0);
        assert_eq!(argmax_token_id(&[-1.0, -2.0, -0.5]).unwrap(), 2);
    }

    #[test]
    fn argmax_empty_returns_error() {
        assert!(argmax_token_id(&[]).is_err());
    }

    #[test]
    fn lm_head_graph_matches_host_sampling() {
        use super::super::generation::greedy_next_token_id;
        use crate::backend::ensure_backends_loaded;
        use ggml_rs::BackendKind;

        let hidden_features = 8_usize;
        let vocab_size = 6_usize;
        let rms_norm_eps = 1e-5_f32;

        let norm_weight: Vec<f32> = (0..hidden_features)
            .map(|i| 0.8 + (i as f32) * 0.05)
            .collect();
        let output_weight: Vec<f32> = (0..hidden_features * vocab_size)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
            .collect();
        let hidden_state: Vec<f32> = (0..hidden_features)
            .map(|i| (i as f32 + 1.0) * 0.2)
            .collect();

        // Host path: rms_norm_with_weight → greedy_next_token_id
        let normed = rms_norm_with_weight(
            &hidden_state,
            hidden_features,
            1,
            &norm_weight,
            rms_norm_eps,
        )
        .expect("host rms_norm");
        let host_token =
            greedy_next_token_id(&normed, 0, hidden_features, &output_weight, vocab_size)
                .expect("host sampling");

        // Graph path
        ensure_backends_loaded();
        let backend = Backend::new(BackendKind::Cpu).expect("CPU backend");
        let graph_logits = lm_head_graph(
            &hidden_state,
            &norm_weight,
            &output_weight,
            hidden_features,
            vocab_size,
            rms_norm_eps,
            &backend,
        )
        .expect("lm_head_graph");
        let graph_token = argmax_token_id(&graph_logits).expect("argmax");

        assert_eq!(
            host_token, graph_token,
            "host token {host_token} != graph token {graph_token}"
        );
    }

    #[test]
    fn lm_head_sample_step_matches_one_shot() {
        use crate::backend::ensure_backends_loaded;
        use ggml_rs::BackendKind;

        let hidden_features = 8_usize;
        let vocab_size = 6_usize;
        let rms_norm_eps = 1e-5_f32;

        let norm_weight: Vec<f32> = (0..hidden_features)
            .map(|i| 0.8 + (i as f32) * 0.05)
            .collect();
        let output_weight: Vec<f32> = (0..hidden_features * vocab_size)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
            .collect();
        let hidden_state: Vec<f32> = (0..hidden_features)
            .map(|i| (i as f32 + 1.0) * 0.2)
            .collect();

        ensure_backends_loaded();
        let backend = Backend::new(BackendKind::Cpu).expect("CPU backend");

        // One-shot graph
        let one_shot_logits = lm_head_graph(
            &hidden_state,
            &norm_weight,
            &output_weight,
            hidden_features,
            vocab_size,
            rms_norm_eps,
            &backend,
        )
        .expect("one-shot");
        let one_shot_token = argmax_token_id(&one_shot_logits).expect("argmax");

        // Persistent graph via build_lm_head_graph
        let ctx_size = recommended_lm_head_memory(hidden_features, vocab_size).expect("mem");
        let ctx = Context::new_no_alloc_bytes(ctx_size).expect("ctx");
        let mut parts =
            build_lm_head_graph(&ctx, hidden_features, vocab_size, rms_norm_eps).expect("build");
        let _buf = ctx.allocate_tensors(&backend).expect("alloc");
        parts
            .w_out
            .write_data_backend(&output_weight)
            .expect("write w_out");
        parts
            .norm_w
            .write_data_backend(&norm_weight)
            .expect("write norm_w");

        let step_token = lm_head_sample_step(
            &hidden_state,
            &parts.x_in,
            &parts.logits,
            &mut parts.graph,
            &backend,
        )
        .expect("sample_step");

        assert_eq!(
            one_shot_token, step_token,
            "one-shot token {one_shot_token} != step token {step_token}"
        );
    }

    #[test]
    fn persistent_full_attention_projection_matches_one_shot() {
        use crate::backend::ensure_backends_loaded;
        use ggml_rs::BackendKind;

        let hidden = 8_usize;
        let qf_x2 = 12_usize; // query_features * 2 (Q+gate interleaved)
        let kv = 4_usize;
        let qf = 6_usize; // query_features (for output proj)

        let w_q: Vec<f32> = (0..hidden * qf_x2)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();
        let w_k: Vec<f32> = (0..hidden * kv)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.08)
            .collect();
        let w_v: Vec<f32> = (0..hidden * kv)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
            .collect();
        let w_out: Vec<f32> = (0..qf * hidden)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
            .collect();
        let input: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.2).collect();

        ensure_backends_loaded();
        let backend = Backend::new(BackendKind::Cpu).expect("CPU backend");

        // One-shot host projection
        let q_host = project_sequence(&input, 1, hidden, qf_x2, &w_q).expect("host Q");
        let k_host = project_sequence(&input, 1, hidden, kv, &w_k).expect("host K");
        let v_host = project_sequence(&input, 1, hidden, kv, &w_v).expect("host V");

        // Persistent projection
        let ctx_size =
            recommended_persistent_full_attention_memory(hidden, qf_x2, kv, qf).expect("mem");
        let ctx = Context::new_no_alloc_bytes(ctx_size).expect("ctx");
        let g = build_persistent_full_attention_graphs(&ctx, hidden, qf_x2, kv, qf).expect("build");
        let _buf = ctx.allocate_tensors(&backend).expect("alloc");

        g.w_q.write_data_backend(&w_q).expect("write W_Q");
        g.w_k.write_data_backend(&w_k).expect("write W_K");
        g.w_v.write_data_backend(&w_v).expect("write W_V");
        g.output.w.write_data_backend(&w_out).expect("write W_OUT");

        let mut proj = PersistentDecodeProjection::FullAttention {
            x_in: g.x_in,
            q_out: g.q_out,
            k_out: g.k_out,
            v_out: g.v_out,
            input_graph: g.input_graph,
            output: g.output,
            _buffer: _buf,
        };
        proj.project_input(&input, &backend).expect("project_input");
        let qkv = proj.read_full_attention_projections().expect("read QKV");

        for (i, (h, p)) in q_host.iter().zip(qkv.q_full.iter()).enumerate() {
            assert!(
                (h - p).abs() < 1e-5,
                "Q[{i}]: host={h} vs persistent={p}, diff={}",
                (h - p).abs()
            );
        }
        for (i, (h, p)) in k_host.iter().zip(qkv.k_proj.iter()).enumerate() {
            assert!(
                (h - p).abs() < 1e-5,
                "K[{i}]: host={h} vs persistent={p}, diff={}",
                (h - p).abs()
            );
        }
        for (i, (h, p)) in v_host.iter().zip(qkv.v_proj.iter()).enumerate() {
            assert!(
                (h - p).abs() < 1e-5,
                "V[{i}]: host={h} vs persistent={p}, diff={}",
                (h - p).abs()
            );
        }

        // Output projection parity
        let core_out: Vec<f32> = (0..qf).map(|i| (i as f32 + 0.5) * 0.3).collect();
        let out_host = project_sequence(&core_out, 1, qf, hidden, &w_out).expect("host out");
        let out_pers = proj
            .project_output(&core_out, &backend)
            .expect("persistent out");

        for (i, (h, p)) in out_host.iter().zip(out_pers.iter()).enumerate() {
            assert!(
                (h - p).abs() < 1e-5,
                "OUT[{i}]: host={h} vs persistent={p}, diff={}",
                (h - p).abs()
            );
        }
    }

    #[test]
    fn persistent_linear_attention_projection_matches_one_shot() {
        use crate::backend::ensure_backends_loaded;
        use ggml_rs::BackendKind;

        let hidden = 8_usize;
        let conv_ch = 10_usize;
        let inner = 6_usize;
        let tsr = 4_usize; // time_step_rank

        let w_qkv: Vec<f32> = (0..hidden * conv_ch)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();
        let w_z: Vec<f32> = (0..hidden * inner)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.08)
            .collect();
        let w_alpha: Vec<f32> = (0..hidden * tsr)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
            .collect();
        let w_beta: Vec<f32> = (0..hidden * tsr)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
            .collect();
        let w_out: Vec<f32> = (0..inner * hidden)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.04)
            .collect();
        let input: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.2).collect();

        ensure_backends_loaded();
        let backend = Backend::new(BackendKind::Cpu).expect("CPU backend");

        // One-shot host projections
        let qkv_host = project_sequence(&input, 1, hidden, conv_ch, &w_qkv).expect("host QKV");
        let z_host = project_sequence(&input, 1, hidden, inner, &w_z).expect("host Z");
        let alpha_host = project_sequence(&input, 1, hidden, tsr, &w_alpha).expect("host alpha");
        let beta_host = project_sequence(&input, 1, hidden, tsr, &w_beta).expect("host beta");

        // Persistent projection
        let ctx_size = recommended_persistent_linear_attention_memory(hidden, conv_ch, inner, tsr)
            .expect("mem");
        let ctx = Context::new_no_alloc_bytes(ctx_size).expect("ctx");
        let g = build_persistent_linear_attention_graphs(&ctx, hidden, conv_ch, inner, tsr)
            .expect("build");
        let _buf = ctx.allocate_tensors(&backend).expect("alloc");

        g.w_qkv.write_data_backend(&w_qkv).expect("write W_QKV");
        g.w_z.write_data_backend(&w_z).expect("write W_Z");
        g.w_alpha
            .write_data_backend(&w_alpha)
            .expect("write W_ALPHA");
        g.w_beta.write_data_backend(&w_beta).expect("write W_BETA");
        g.output.w.write_data_backend(&w_out).expect("write W_OUT");

        let mut proj = PersistentDecodeProjection::LinearAttention {
            x_in: g.x_in,
            qkv_out: g.qkv_out,
            z_out: g.z_out,
            alpha_out: g.alpha_out,
            beta_out: g.beta_out,
            input_graph: g.input_graph,
            output: g.output,
            _buffer: _buf,
        };
        proj.project_input(&input, &backend).expect("project_input");
        let raw = proj
            .read_linear_attention_projections()
            .expect("read linear");

        for (name, host, pers) in [
            ("QKV", &qkv_host, &raw.qkv),
            ("Z", &z_host, &raw.z),
            ("alpha", &alpha_host, &raw.alpha),
            ("beta", &beta_host, &raw.beta),
        ] {
            assert_eq!(host.len(), pers.len(), "{name} length mismatch");
            for (i, (h, p)) in host.iter().zip(pers.iter()).enumerate() {
                assert!(
                    (h - p).abs() < 1e-5,
                    "{name}[{i}]: host={h} vs persistent={p}, diff={}",
                    (h - p).abs()
                );
            }
        }

        // Output projection parity
        let core_out: Vec<f32> = (0..inner).map(|i| (i as f32 + 0.5) * 0.3).collect();
        let out_host = project_sequence(&core_out, 1, inner, hidden, &w_out).expect("host out");
        let out_pers = proj
            .project_output(&core_out, &backend)
            .expect("persistent out");

        for (i, (h, p)) in out_host.iter().zip(out_pers.iter()).enumerate() {
            assert!(
                (h - p).abs() < 1e-5,
                "OUT[{i}]: host={h} vs persistent={p}, diff={}",
                (h - p).abs()
            );
        }
    }
}
