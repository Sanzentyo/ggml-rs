use super::InferenceError;
use crate::backend::{LlamaBackend, ensure_backends_loaded};
use ggml_rs::{Backend, Bytes, Context};

pub(super) struct BackendRuntime {
    pub(super) backend: Backend,
    pub(super) backend_name: String,
    pub(super) ctx: Context,
}

pub(super) trait BackendRuntimeBuilder {
    fn build_runtime(
        &self,
        backend_kind: LlamaBackend,
        ctx_size: Bytes,
    ) -> Result<BackendRuntime, InferenceError>;
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct DefaultBackendRuntimeBuilder;

impl BackendRuntimeBuilder for DefaultBackendRuntimeBuilder {
    fn build_runtime(
        &self,
        backend_kind: LlamaBackend,
        ctx_size: Bytes,
    ) -> Result<BackendRuntime, InferenceError> {
        ensure_backends_loaded();
        let backend = Backend::new(backend_kind.into())
            .map_err(|source| InferenceError::ggml("Backend::new", source))?;
        let backend_name = backend
            .name()
            .map_err(|source| InferenceError::ggml("Backend::name", source))?
            .to_string();
        let ctx = Context::new_no_alloc_bytes(ctx_size)
            .map_err(|source| InferenceError::ggml("Context::new_no_alloc_bytes", source))?;
        Ok(BackendRuntime {
            backend,
            backend_name,
            ctx,
        })
    }
}
