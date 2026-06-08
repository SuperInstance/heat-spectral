use crate::error::{HeatError, Result};
use crate::graph::Graph;
use crate::kernel::HeatKernel;

/// Apply a heat kernel as a spectral low-pass filter to denoise a graph signal.
///
/// This computes `exp(-τ L) x`, smoothing out high-frequency (rapidly varying)
/// components of the signal `x` on the graph.
pub fn spectral_filter(
    graph: &mut Graph,
    signal: &[f64],
    tau: f64,
    order: usize,
) -> Result<Vec<f64>> {
    let n = graph.node_count();
    if signal.len() != n {
        return Err(HeatError::SignalMismatch { len: signal.len(), n });
    }
    if tau < 0.0 {
        return Err(HeatError::NegativeTau { tau });
    }
    if order == 0 {
        return Err(HeatError::InvalidOrder { order });
    }

    let mut kernel = HeatKernel::new(graph.clone(), tau, order)?;
    let filtered = kernel.apply(signal)?;
    Ok(filtered)
}

/// Multi-scale spectral filtering: apply heat kernels at multiple τ values
/// and aggregate the results (average).
pub fn multiscale_filter(
    graph: &mut Graph,
    signal: &[f64],
    taus: &[f64],
    order: usize,
) -> Result<Vec<f64>> {
    let n = graph.node_count();
    if taus.is_empty() {
        return Ok(signal.to_vec());
    }

    let mut acc = vec![0.0; n];
    for &tau in taus {
        let filtered = spectral_filter(graph, signal, tau, order)?;
        for i in 0..n {
            acc[i] += filtered[i];
        }
    }
    for x in &mut acc {
        *x /= taus.len() as f64;
    }
    Ok(acc)
}
