use thiserror::Error;

/// Errors produced by the heat-spectral crate.
#[derive(Debug, Error)]
pub enum HeatError {
    #[error("invalid node index {index} for graph with {n} nodes")]
    InvalidNode { index: usize, n: usize },
    #[error("graph must have at least 2 nodes for Fiedler value computation")]
    TooFewNodes,
    #[error("graph is disconnected; Fiedler value is zero")]
    Disconnected,
    #[error("diffusion time parameter tau must be non-negative, got {tau}")]
    NegativeTau { tau: f64 },
    #[error("Chebyshev approximation order must be >= 1, got {order}")]
    InvalidOrder { order: usize },
    #[error("signal length {len} does not match graph size {n}")]
    SignalMismatch { len: usize, n: usize },
}

pub type Result<T> = std::result::Result<T, HeatError>;
