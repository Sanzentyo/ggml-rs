use ggml_rs::BackendKind;
use std::error::Error as StdError;
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LlamaBackend {
    Cpu,
    Metal,
}

impl LlamaBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }
}

impl From<LlamaBackend> for BackendKind {
    fn from(value: LlamaBackend) -> Self {
        match value {
            LlamaBackend::Cpu => BackendKind::Cpu,
            LlamaBackend::Metal => BackendKind::Metal,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseBackendError {
    input: String,
}

impl ParseBackendError {
    fn new(input: &str) -> Self {
        Self {
            input: input.to_string(),
        }
    }
}

impl fmt::Display for ParseBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown backend `{}` (expected `cpu` or `metal`)",
            self.input
        )
    }
}

impl StdError for ParseBackendError {}

impl FromStr for LlamaBackend {
    type Err = ParseBackendError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            _ => Err(ParseBackendError::new(s)),
        }
    }
}
