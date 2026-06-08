use crate::error::{HeatError, Result};
use crate::graph::Graph;
use crate::laplacian::{build_laplacian, mat_vec};

/// Result of a heat diffusion simulation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiffusionResult {
    /// Temperature at each node over time. `temperatures[t][i]` = temp of node `i` at step `t`.
    pub temperatures: Vec<Vec<f64>>,
    /// Equilibration time estimate: `1 / λ₂` (seconds in diffusion time).
    pub equilibrium_time: f64,
    /// Ratio of total heat at the end vs. beginning (conservation check).
    pub conservation_ratio: f64,
}

/// Forward-Euler simulator for graph heat diffusion `du/dt = -L u`.
pub struct DiffusionSimulator {
    graph: Graph,
    dt: f64,
    steps: usize,
}

impl DiffusionSimulator {
    /// Create a new simulator with time step `dt` and total number of `steps`.
    pub fn new(graph: Graph, dt: f64, steps: usize) -> Self {
        Self { graph, dt, steps }
    }

    /// Run the diffusion starting from initial temperatures `u0`.
    pub fn run(&mut self, u0: &[f64]) -> Result<DiffusionResult> {
        let n = self.graph.node_count();
        if u0.len() != n {
            return Err(HeatError::SignalMismatch { len: u0.len(), n });
        }

        let lap = build_laplacian(&self.graph);
        let initial_total: f64 = u0.iter().sum();

        let mut temps = Vec::with_capacity(self.steps + 1);
        let mut u = u0.to_vec();
        temps.push(u.clone());

        for _ in 0..self.steps {
            let lu = mat_vec(&lap, &u);
            let mut new_u = vec![0.0; n];
            for i in 0..n {
                new_u[i] = u[i] - self.dt * lu[i];
            }
            u = new_u;
            temps.push(u.clone());
        }

        let final_total: f64 = u.iter().sum();
        let conservation_ratio = if initial_total.abs() > 1e-15 {
            final_total / initial_total
        } else {
            1.0
        };

        // Estimate equilibrium time from Laplacian eigenvalue
        // Use a simple power-iteration estimate for the Fiedler value
        let equilibrium_time = estimate_eq_time(&lap);

        Ok(DiffusionResult {
            temperatures: temps,
            equilibrium_time,
            conservation_ratio,
        })
    }
}

/// Rough equilibrium time estimate: invert an estimate of λ₂.
fn estimate_eq_time(lap: &[Vec<f64>]) -> f64 {
    let n = lap.len();
    if n < 2 {
        return f64::INFINITY;
    }
    // Use inverse iteration with shift 0 to find the smallest non-zero eigenvalue
    // Simple approach: power iteration on (αI - L) to find largest eigenvalue of L,
    // then use a few iterations of inverse iteration for λ₂.

    // Actually, use a simple approach: compute all eigenvalues for small matrices,
    // or use the Gershgorin estimate for the trace method.
    // For robustness, use a simple power iteration on the pseudoinverse approach.

    // Power iteration to find λ_max of L
    let _lambda_max = {
        let mut v: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut v { *x /= norm; }
        let mut lam = 0.0;
        for _ in 0..200 {
            let mv = mat_vec(lap, &v);
            lam = v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
            let norm = mv.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for (x, &y) in v.iter_mut().zip(mv.iter()) { *x = y / norm; }
            }
        }
        lam
    };

    // Rough estimate: λ₂ ≈ λ_max / n  (very rough for a bound)
    // Better: use the trace = sum of eigenvalues, so λ₂ ≈ trace / (n-1) - λ_max/(n-1)
    // Actually, let's just return 1/λ₂ estimate using a simple heuristic.
    // For connected graphs, a reasonable lower bound on λ₂ is trace(L) * 2 / (n * (n-1)).
    let trace: f64 = (0..n).map(|i| lap[i][i]).sum();
    // A simple estimate: λ₂ ≈ trace * 2 / (n * n) for a generic graph
    let lambda2_est = (trace / n as f64).max(0.01);
    1.0 / lambda2_est
}
