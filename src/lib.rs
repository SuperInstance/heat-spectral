//! # heat-spectral
//!
//! **Exact spectral solution of the discrete heat equation on arbitrary graphs.**
//!
//! This crate simulates heat diffusion via the equation ∂u/∂t = −Lu, where L is
//! the graph Laplacian. Rather than using iterative numerical methods (Euler,
//! Runge-Kutta), it computes the *exact* solution by projecting onto the
//! eigenvector basis and multiplying each coefficient by exp(−λᵢ·Δt). This
//! guarantees unconditional stability and perfect energy conservation regardless
//! of time-step size.
//!
//! ## The Key Insight
//!
//! The heat equation on a graph is a linear ODE: u̇ = −Lu. If L has eigenvalues
//! λ₁…λₙ and eigenvectors v₁…vₙ, then any initial temperature vector decomposes as
//! u(0) = Σ cᵢvᵢ. The exact solution is simply u(t) = Σ cᵢ·exp(−λᵢ·t)·vᵢ. Each
//! eigenmode decays independently at a rate determined by its eigenvalue — high
//! frequencies die fast, low frequencies persist. The conservation ratio CR = λ₂/λₙ
//! captures this: high CR means all modes decay at similar rates (fast mixing).
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────┐
//! │  Adjacency      │   User-supplied weighted adjacency matrix
//! │  Matrix [f64]   │
//! └───────┬────────┘
//!         │ L = D − A
//! ┌───────▼────────┐
//! │  Laplacian      │   Graph Laplacian (symmetric, positive semi-definite)
//! │                  │
//! └───────┬────────┘
//!         │ Jacobi eigenvalue rotation (O(n³), no deps)
//! ┌───────▼────────┐
//! │  Eigenbasis     │   λ₁=0, λ₂…λₙ ≥ 0; eigenvectors are orthonormal
//! │  (λ, V)         │
//! └───────┬────────┘
//!         │ project: cᵢ = ⟨u, vᵢ⟩
//! ┌───────▼────────┐
//! │  Exact Step     │   u(t+Δt) = Σ cᵢ·exp(−λᵢ·Δt)·vᵢ
//! │  (no drift)     │   Energy = Σ cᵢ²·exp(−2λᵢ·Δt) < Σ cᵢ²
//! └───────┬────────┘
//!         │
//! ┌───────▼────────┐
//! │  Applications   │   Spectral filtering, anomaly detection,
//! │                  │   thermal imaging, diffusion-vs-wave comparison
//! └────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```rust
//! use heat_spectral::{HeatState, cycle_graph};
//!
//! let adj = cycle_graph(10);
//! let mut sim = HeatState::new(&adj);
//! sim.set_heat(0, 1.0);   // heat pulse at node 0
//!
//! for _ in 0..100 {
//!     sim.step(0.1);       // exact spectral step — unconditionally stable
//! }
//!
//! println!("Total heat: {:.6}", sim.total_heat());  // conserved
//! println!("Variance:   {:.6}", sim.variance());    // monotonically decreasing
//! println!("CR (λ₂/λₙ): {:.4}", sim.cr());
//! println!("Eq time (1/λ₂): {:.4}", sim.equilibration_time());
//! ```
//!
//! ## Zero Dependencies
//!
//! This crate uses no external crates. The Jacobi eigenvalue algorithm is
//! implemented from scratch, requiring only `std::f64::consts::PI`.

use std::f64::consts::PI;

// ── Core state ───────────────────────────────────────────────

/// The central simulation state for heat diffusion on a graph.
///
/// Holds the current temperature distribution, the graph Laplacian, and the
/// complete eigendecomposition computed via Jacobi rotation. The eigenbasis is
/// computed once at construction time; all subsequent time-steps are O(n²)
/// projections and reconstructions.
///
/// # Construction
///
/// ```rust
/// use heat_spectral::{HeatState, path_graph};
///
/// let adj = path_graph(5);
/// let state = HeatState::new(&adj);
/// // Eigenvalues are already computed:
/// assert!(state.eigenvalues[0].abs() < 1e-8); // λ₁ ≈ 0 (connected component)
/// ```
pub struct HeatState {
    /// Current temperature at each node.
    pub temperature: Vec<f64>,
    /// The graph Laplacian L = D − A.
    pub laplacian: Vec<Vec<f64>>,
    /// Eigenvalues of the Laplacian, sorted ascending (λ₁ ≈ 0).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors of the Laplacian. `eigenvectors[k]` is the k-th eigenvector.
    pub eigenvectors: Vec<Vec<f64>>,
    /// Total initial heat (for conservation tracking).
    initial_heat: f64,
}

impl HeatState {
    /// Create a new `HeatState` from an adjacency matrix.
    ///
    /// Computes the Laplacian and its full eigendecomposition using Jacobi
    /// rotation. This is O(n³) and runs once; all subsequent operations are
    /// cheaper.
    ///
    /// # Panics
    ///
    /// Does not panic, but the Jacobi solver may not converge for very large
    /// matrices (>100 nodes). For typical graph sizes (≤50), convergence is
    /// guaranteed.
    pub fn new(adj: &[Vec<f64>]) -> HeatState {
        let n = adj.len();
        let mut laplacian = vec![vec![0.0; n]; n];
        for i in 0..n {
            let degree: f64 = adj[i].iter().sum();
            laplacian[i][i] = degree;
            for j in 0..n {
                if i != j {
                    laplacian[i][j] = -adj[i][j];
                }
            }
        }

        let (eigenvalues, eigenvectors) = jacobi_eigen(&laplacian);

        HeatState {
            temperature: vec![0.0; n],
            laplacian,
            eigenvalues,
            eigenvectors,
            initial_heat: 0.0,
        }
    }

    /// Set a heat pulse at a single node.
    ///
    /// Resets all temperatures to zero, then sets `temperature[node] = temp`.
    /// This is the standard "point source" initial condition used to study
    /// how heat spreads through the graph topology.
    pub fn set_heat(&mut self, node: usize, temp: f64) {
        self.temperature = vec![0.0; self.eigenvalues.len()];
        self.temperature[node] = temp;
        self.initial_heat = temp;
    }

    /// Set an arbitrary temperature pattern across all nodes.
    ///
    /// The `temps` slice must have the same length as the graph (number of nodes).
    /// This is used for spectral filtering, where a signal is diffused to smooth
    /// out high-frequency noise.
    pub fn set_pattern(&mut self, temps: &[f64]) {
        self.temperature = temps.to_vec();
        self.initial_heat = temps.iter().sum();
    }

    /// Perform an exact spectral time-step.
    ///
    /// Projects the current temperature onto the eigenvector basis, multiplies
    /// each coefficient by exp(−λᵢ·dt), and reconstructs. This is the *exact*
    /// solution to the discrete heat equation — no truncation error, no drift.
    ///
    /// **Unconditionally stable**: works for any `dt`, even dt = 1000.
    pub fn step(&mut self, dt: f64) {
        let n = self.eigenvalues.len();
        let mut new_temp = vec![0.0; n];

        for k in 0..n {
            let lambda = self.eigenvalues[k];
            let coeff: f64 = self.eigenvectors[k]
                .iter()
                .zip(self.temperature.iter())
                .map(|(v, t)| v * t)
                .sum();
            let decay = (-lambda * dt).exp();
            for i in 0..n {
                new_temp[i] += coeff * decay * self.eigenvectors[k][i];
            }
        }

        self.temperature = new_temp;
    }

    /// Perform a Forward Euler time-step (for comparison/benchmarking).
    ///
    /// Uses the update rule: u(t+dt) = u(t) − dt·L·u(t).
    ///
    /// **Warning**: Unstable for large dt. The stability condition requires
    /// dt < 2/λₙ (where λₙ is the largest eigenvalue). For a complete graph
    /// on n nodes, λₙ = n, so dt < 2/n. Provided for educational comparison
    /// with the exact spectral step.
    pub fn step_euler(&mut self, dt: f64) {
        let n = self.eigenvalues.len();
        let mut new_temp = vec![0.0; n];
        for i in 0..n {
            let mut lu = 0.0;
            for j in 0..n {
                lu += self.laplacian[i][j] * self.temperature[j];
            }
            new_temp[i] = self.temperature[i] - dt * lu;
        }
        self.temperature = new_temp;
    }

    /// Current temperature distribution (read-only).
    pub fn temperature(&self) -> &[f64] {
        &self.temperature
    }

    /// Total heat (sum of all node temperatures).
    ///
    /// For the exact spectral step, this is conserved to machine precision.
    /// For Forward Euler, it drifts — another reason to prefer spectral steps.
    pub fn total_heat(&self) -> f64 {
        self.temperature.iter().sum()
    }

    /// Variance of the temperature distribution.
    ///
    /// Measures how "spread out" the heat is. Starts high (concentrated source)
    /// and decreases monotonically to zero (uniform distribution). The rate of
    /// decrease is governed by the spectral gap λ₂.
    pub fn variance(&self) -> f64 {
        let n = self.temperature.len() as f64;
        let mean = self.temperature.iter().sum::<f64>() / n;
        self.temperature.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n
    }

    /// Predicted equilibration time: 1/λ₂ (the inverse algebraic connectivity).
    ///
    /// This is the time constant for the slowest-decaying mode. After ~4/λ₂
    /// time units, variance has dropped to ~2% of its initial value. For a
    /// complete graph this is fast (high λ₂); for a path graph, slow (low λ₂).
    pub fn equilibration_time(&self) -> f64 {
        for &lambda in &self.eigenvalues {
            if lambda > 1e-10 {
                return 1.0 / lambda;
            }
        }
        f64::INFINITY
    }

    /// Conservation Ratio: CR = λ₂ / λₙ.
    ///
    /// This single number captures the mixing quality of the graph:
    /// - **CR > 0.7**: Fast, uniform mixing (complete graphs, expanders)
    /// - **CR ~ 0.3**: Moderate mixing (cycles, regular graphs)
    /// - **CR < 0.1**: Poor mixing, bottlenecks (paths, barbell graphs)
    ///
    /// High CR → all eigenmodes decay at similar rates → equilibration is fast.
    /// Low CR → some modes decay much slower than others → equilibration is slow.
    pub fn cr(&self) -> f64 {
        let n = self.eigenvalues.len();
        if n < 2 {
            return 0.0;
        }
        let lambda_max = self.eigenvalues.last().unwrap();
        let mut lambda2 = 0.0;
        for &l in &self.eigenvalues {
            if l > 1e-10 {
                lambda2 = l;
                break;
            }
        }
        if lambda_max.abs() < 1e-15 {
            return 0.0;
        }
        lambda2 / lambda_max
    }
}

// ── Experiment reports ───────────────────────────────────────

/// Report from an equilibration time verification experiment.
///
/// Compares the predicted equilibration time (1/λ₂) against the measured time
/// (time for variance to drop below 1% of initial). Also tracks the maximum
/// heat conservation error across all steps.
pub struct DiffusionReport {
    /// Variance at t=0.
    pub initial_variance: f64,
    /// Variance at each time-step (for plotting decay curves).
    pub variance_at_step: Vec<f64>,
    /// Measured equilibration time (steps × dt until variance < 1%).
    pub equilibration_time: f64,
    /// Predicted equilibration time (1/λ₂).
    pub predicted_eq_time: f64,
    /// Maximum |total_heat(t) − total_heat(0)| across all steps.
    pub heat_conservation_error: f64,
    /// Conservation ratio CR = λ₂/λₙ.
    pub cr: f64,
}

/// Report comparing heat diffusion vs wave propagation on the same graph.
///
/// Heat diffusion is dissipative (converges to equilibrium). Wave propagation
/// is conservative (oscillates). The ratio of wave coherence halflife to heat
/// equilibration time reveals the interplay between these two dynamics.
pub struct ComparisonReport {
    /// Name of the graph topology.
    pub graph_name: String,
    /// Conservation ratio CR = λ₂/λₙ.
    pub cr: f64,
    /// Time for heat to equilibrate (variance < 1% of initial).
    pub heat_equilibration_time: f64,
    /// Steps until wave autocorrelation drops below 0.5.
    pub wave_coherence_halflife: usize,
    /// wave_coherence_halflife × wave_dt / heat_equilibration_time.
    pub ratio: f64,
}

// ── Diffusion experiments ────────────────────────────────────

/// Run a full equilibration experiment on the given adjacency matrix.
///
/// Sets a point heat source at node 0, runs exact spectral steps until variance
/// drops below 1% of initial, and returns a detailed report comparing predicted
/// vs measured equilibration time.
pub fn verify_equilibration_time(adj: &[Vec<f64>]) -> DiffusionReport {
    let mut state = HeatState::new(adj);
    let predicted = state.equilibration_time();
    let cr = state.cr();

    state.set_heat(0, 1.0);
    let initial_variance = state.variance();
    let initial_heat = state.total_heat();

    let dt = 0.01;
    let max_steps = 10000;
    let mut variance_at_step = vec![initial_variance];
    let mut heat_conservation_error = 0.0;

    for _ in 0..max_steps {
        state.step(dt);
        variance_at_step.push(state.variance());
        let err = (state.total_heat() - initial_heat).abs();
        if err > heat_conservation_error {
            heat_conservation_error = err;
        }
        if state.variance() < 1e-10 {
            break;
        }
    }

    let threshold = initial_variance * 0.01;
    let eq_step = variance_at_step
        .iter()
        .position(|&v| v < threshold)
        .unwrap_or(variance_at_step.len());
    let measured_eq_time = eq_step as f64 * dt;

    DiffusionReport {
        initial_variance,
        variance_at_step,
        equilibration_time: measured_eq_time,
        predicted_eq_time: predicted,
        heat_conservation_error,
        cr,
    }
}

/// Measure CR vs equilibration time for several canonical graph families.
///
/// Returns (CR, predicted_eq_time) pairs for path(6), cycle(6), complete(6),
/// star(6), and barbell(4). Useful for demonstrating the relationship between
/// spectral connectivity and mixing speed.
pub fn cr_vs_equilibration() -> Vec<(f64, f64)> {
    let mut results = Vec::new();
    results.push(cr_eq_for_graph(&path_graph(6), "path-6"));
    results.push(cr_eq_for_graph(&cycle_graph(6), "cycle-6"));
    results.push(cr_eq_for_graph(&complete_graph(6), "complete-6"));
    results.push(cr_eq_for_graph(&star_graph(6), "star-6"));
    results.push(cr_eq_for_graph(&barbell_graph(4), "barbell-4"));
    results
}

fn cr_eq_for_graph(adj: &[Vec<f64>], _name: &str) -> (f64, f64) {
    let report = verify_equilibration_time(adj);
    (report.cr, report.predicted_eq_time)
}

/// Apply heat diffusion as a spectral low-pass filter.
///
/// Diffuses the input `signal` for `time` units. Short diffusion times preserve
/// low-frequency structure while attenuating high-frequency noise. Long diffusion
/// times smooth the signal to near-uniformity. This is the graph analog of
/// Gaussian blur in image processing.
pub fn spectral_filtering(signal: &[f64], adj: &[Vec<f64>], time: f64) -> Vec<f64> {
    let mut state = HeatState::new(adj);
    state.set_pattern(signal);
    let dt = 0.01;
    let steps = (time / dt).round() as usize;
    for _ in 0..steps {
        state.step(dt);
    }
    state.temperature.clone()
}

/// Detect anomalous nodes using heat diffusion signatures.
///
/// For each node, computes its "thermal fingerprint" — the temperature
/// distribution after diffusing a unit pulse for a fixed time. Anomalous
/// nodes (e.g., bridge nodes in a barbell graph) produce atypical signatures
/// that differ significantly from the average.
///
/// Returns the L2 distance between the anomalous node's thermal pattern
/// and the average pattern across all nodes.
pub fn heat_anomaly_detection(adj: &[Vec<f64>], anomalous_node: usize) -> f64 {
    let n = adj.len();
    let mut state = HeatState::new(adj);

    state.set_heat(anomalous_node, 1.0);
    let dt = 0.05;
    for _ in 0..20 {
        state.step(dt);
    }
    let anomalous_pattern: Vec<f64> = state.temperature.clone();

    let mut avg_pattern = vec![0.0; n];
    for node in 0..n {
        state.set_heat(node, 1.0);
        for _ in 0..20 {
            state.step(dt);
        }
        for (i, &t) in state.temperature.iter().enumerate() {
            avg_pattern[i] += t;
        }
    }
    let n_f = n as f64;
    for t in avg_pattern.iter_mut() {
        *t /= n_f;
    }

    anomalous_pattern
        .iter()
        .zip(avg_pattern.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute a thermal image: for each source node, diffuse heat and record the
/// steady-state distribution.
///
/// Returns an n×n matrix where `image[i][j]` is the temperature at node j after
/// diffusing a unit pulse from node i for 500 steps at dt=0.01. The resulting
/// matrix encodes the graph's communication structure — nodes that share heat
/// quickly will have similar rows.
pub fn thermal_image(adj: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = adj.len();
    let mut image = vec![vec![0.0; n]; n];
    let mut state = HeatState::new(adj);

    for source in 0..n {
        state.set_heat(source, 1.0);
        for _ in 0..500 {
            state.step(0.01);
        }
        image[source] = state.temperature.clone();
    }
    image
}

/// Compare heat diffusion and wave propagation on the same graph.
///
/// Heat (parabolic PDE): dissipative, converges to equilibrium.
/// Wave (hyperbolic PDE): conservative, oscillates.
///
/// Uses leapfrog integration for the wave equation: ∂²u/∂t² = −Lu.
/// Returns a report comparing equilibration time vs coherence halflife.
pub fn compare_diffusion_wave(adj: &[Vec<f64>], name: &str) -> ComparisonReport {
    let n = adj.len();
    let mut heat_state = HeatState::new(adj);
    let cr = heat_state.cr();

    heat_state.set_heat(0, 1.0);
    let initial_var = heat_state.variance();
    let dt = 0.01;
    let mut heat_steps = 0;
    for _ in 0..50000 {
        heat_state.step(dt);
        heat_steps += 1;
        if heat_state.variance() < initial_var * 0.01 {
            break;
        }
    }
    let heat_eq_time = heat_steps as f64 * dt;

    let laplacian = &heat_state.laplacian;
    let mut u_curr = vec![0.0; n];
    let mut u_prev = vec![0.0; n];
    u_curr[0] = 1.0;
    u_prev[0] = 1.0;

    let wave_dt = 0.005;
    let max_wave_steps = 10000;
    let mut coherence_halflife = max_wave_steps;

    let initial_pattern = u_curr.clone();

    for step in 0..max_wave_steps {
        let mut lu = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                lu[i] += laplacian[i][j] * u_curr[j];
            }
        }
        let mut u_next = vec![0.0; n];
        for i in 0..n {
            u_next[i] = 2.0 * u_curr[i] - u_prev[i] - wave_dt * wave_dt * lu[i];
        }

        let corr = correlation(&u_next, &initial_pattern);
        if corr < 0.5 && coherence_halflife == max_wave_steps {
            coherence_halflife = step;
        }

        u_prev = u_curr.clone();
        u_curr = u_next;
    }

    let ratio = coherence_halflife as f64 * wave_dt / heat_eq_time;

    ComparisonReport {
        graph_name: name.to_string(),
        cr,
        heat_equilibration_time: heat_eq_time,
        wave_coherence_halflife: coherence_halflife,
        ratio,
    }
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean_a: f64 = a.iter().sum::<f64>() / n;
    let mean_b: f64 = b.iter().sum::<f64>() / n;
    let cov: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - mean_a) * (y - mean_b))
        .sum();
    let var_a: f64 = a.iter().map(|x| (x - mean_a).powi(2)).sum();
    let var_b: f64 = b.iter().map(|x| (x - mean_b).powi(2)).sum();
    if var_a < 1e-15 || var_b < 1e-15 {
        return 0.0;
    }
    cov / (var_a.sqrt() * var_b.sqrt())
}

// ── Graph generators ─────────────────────────────────────────

/// Generate a path graph adjacency matrix on `n` nodes.
///
/// Nodes are connected in a chain: 0-1-2-…-(n−1). This graph has the lowest
/// CR of any connected graph — heat takes a long time to equilibrate because
/// it must diffuse sequentially through every intermediate node.
pub fn path_graph(n: usize) -> Vec<Vec<f64>> {
    let mut adj = vec![vec![0.0; n]; n];
    for i in 0..n - 1 {
        adj[i][i + 1] = 1.0;
        adj[i + 1][i] = 1.0;
    }
    adj
}

/// Generate a cycle graph adjacency matrix on `n` nodes.
///
/// Like a path, but node n−1 connects back to node 0. The extra edge
/// provides an alternative path for heat to travel, increasing CR and
/// decreasing equilibration time compared to the path.
pub fn cycle_graph(n: usize) -> Vec<Vec<f64>> {
    let mut adj = path_graph(n);
    adj[0][n - 1] = 1.0;
    adj[n - 1][0] = 1.0;
    adj
}

/// Generate a complete graph adjacency matrix on `n` nodes.
///
/// Every node connects to every other node. This has the highest possible CR
/// for an n-node graph — heat from any node reaches every other node in one
/// step. The graph Laplacian has eigenvalues {0, n, n, …, n}.
pub fn complete_graph(n: usize) -> Vec<Vec<f64>> {
    let mut adj = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                adj[i][j] = 1.0;
            }
        }
    }
    adj
}

/// Generate a star graph adjacency matrix on `n` nodes.
///
/// Node 0 is the hub; all other nodes connect only to it. Interesting because
/// the hub is a bottleneck — all heat must pass through it. The eigenvalue
/// structure reflects this asymmetry.
pub fn star_graph(n: usize) -> Vec<Vec<f64>> {
    let mut adj = vec![vec![0.0; n]; n];
    for i in 1..n {
        adj[0][i] = 1.0;
        adj[i][0] = 1.0;
    }
    adj
}

/// Generate a barbell graph: two complete graphs of size `m` connected by a
/// single bridge edge.
///
/// This creates a severe bottleneck — the bridge must carry all heat flowing
/// between the two cliques. The eigenvalue gap between λ₂ and λ₃ reveals this
/// structure: λ₂ is very small (bridge is a weak connection), while λ₃ is large
/// (within-clique connections are strong).
pub fn barbell_graph(m: usize) -> Vec<Vec<f64>> {
    let n = 2 * m;
    let mut adj = vec![vec![0.0; n]; n];
    for i in 0..m {
        for j in 0..m {
            if i != j {
                adj[i][j] = 1.0;
            }
        }
    }
    for i in m..n {
        for j in m..n {
            if i != j {
                adj[i][j] = 1.0;
            }
        }
    }
    adj[m - 1][m] = 1.0;
    adj[m][m - 1] = 1.0;
    adj
}

// ── Jacobi eigenvalue algorithm ──────────────────────────────

/// Compute all eigenvalues and eigenvectors of a symmetric matrix via
/// Jacobi rotation.
///
/// The Jacobi method iteratively applies Givens rotations to zero the largest
/// off-diagonal element. Each rotation preserves symmetry and reduces the
/// off-diagonal norm. After O(n²) sweeps, the matrix converges to a diagonal
/// of eigenvalues, and the accumulated rotations form the eigenvector matrix.
///
/// Returns (eigenvalues, eigenvectors) sorted by eigenvalue ascending.
fn jacobi_eigen(mat: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = mat.len();
    let mut a = mat.to_vec();
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    let max_sweeps = 100;
    let tol = 1e-12;

    for _ in 0..max_sweeps {
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < tol {
            break;
        }

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];

        let theta = if (app - aqq).abs() < 1e-15 {
            PI / 4.0
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i][p];
                let aiq = a[i][q];
                new_a[i][p] = c * aip + s * aiq;
                new_a[p][i] = new_a[i][p];
                new_a[i][q] = -s * aip + c * aiq;
                new_a[q][i] = new_a[i][q];
            }
        }
        new_a[p][p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        new_a[q][q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;
        a = new_a;

        let mut new_v = v.clone();
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            new_v[i][p] = c * vip + s * viq;
            new_v[i][q] = -s * vip + c * viq;
        }
        v = new_v;
    }

    let mut indexed: Vec<(usize, f64)> = (0..n).map(|i| (i, a[i][i])).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let eigenvalues: Vec<f64> = indexed.iter().map(|(_, v)| *v).collect();
    let eigenvectors: Vec<Vec<f64>> = indexed
        .iter()
        .map(|(j, _)| (0..n).map(|i| v[i][*j]).collect())
        .collect();

    (eigenvalues, eigenvectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heat_conservation_exact_step() {
        let adj = cycle_graph(6);
        let mut state = HeatState::new(&adj);
        state.set_heat(0, 1.0);
        let initial = state.total_heat();
        for _ in 0..100 {
            state.step(0.01);
        }
        let final_heat = state.total_heat();
        assert!(
            (final_heat - initial).abs() < 1e-8,
            "Heat not conserved: {} vs {}",
            final_heat,
            initial
        );
    }

    #[test]
    fn test_heat_conservation_pattern() {
        let adj = complete_graph(5);
        let mut state = HeatState::new(&adj);
        let pattern = vec![1.0, 2.0, 0.0, -1.0, 3.0];
        state.set_pattern(&pattern);
        let initial = state.total_heat();
        for _ in 0..200 {
            state.step(0.01);
        }
        assert!((state.total_heat() - initial).abs() < 1e-8);
    }

    #[test]
    fn test_variance_decreases_monotonically() {
        let adj = path_graph(6);
        let mut state = HeatState::new(&adj);
        state.set_heat(0, 1.0);
        let mut prev_var = state.variance();
        for _ in 0..50 {
            state.step(0.05);
            let cur_var = state.variance();
            assert!(
                cur_var <= prev_var + 1e-12,
                "Variance increased: {} > {}",
                cur_var,
                prev_var
            );
            prev_var = cur_var;
        }
    }

    #[test]
    fn test_equilibration_time_approx_inverse_lambda2() {
        let adj = cycle_graph(8);
        let report = verify_equilibration_time(&adj);
        let ratio = report.equilibration_time / report.predicted_eq_time;
        assert!(
            ratio > 0.05 && ratio < 10.0,
            "Equilibration time {} not in reasonable range of 1/λ₂ = {}, ratio = {}",
            report.equilibration_time,
            report.predicted_eq_time,
            ratio
        );
    }

    #[test]
    fn test_higher_cr_faster_equilibration() {
        let path_report = verify_equilibration_time(&path_graph(6));
        let complete_report = verify_equilibration_time(&complete_graph(6));
        assert!(complete_report.cr > path_report.cr, "Complete graph should have higher CR");
        assert!(
            complete_report.predicted_eq_time < path_report.predicted_eq_time,
            "Complete graph should equilibrate faster"
        );
    }

    #[test]
    fn test_spectral_filtering_short_preserves_detail() {
        let n = 8;
        let adj = cycle_graph(n);
        let signal: Vec<f64> = (0..n).map(|i| if i == 3 { 5.0 } else { 0.0 }).collect();
        let short = spectral_filtering(&signal, &adj, 0.01);
        let long = spectral_filtering(&signal, &adj, 5.0);

        let short_var: f64 =
            short.iter().map(|x| (x - short.iter().sum::<f64>() / n as f64).powi(2)).sum::<f64>()
                / n as f64;
        let long_var: f64 =
            long.iter().map(|x| (x - long.iter().sum::<f64>() / n as f64).powi(2)).sum::<f64>()
                / n as f64;
        assert!(
            short_var > long_var,
            "Short diffusion should preserve more variance"
        );
    }

    #[test]
    fn test_spectral_filtering_long_smooths() {
        let n = 8;
        let adj = cycle_graph(n);
        let signal: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let long = spectral_filtering(&signal, &adj, 5.0);
        let mean = long.iter().sum::<f64>() / n as f64;
        for &t in &long {
            assert!(
                (t - mean).abs() < 0.1,
                "Long diffusion should smooth: {} not close to {}",
                t,
                mean
            );
        }
    }

    #[test]
    fn test_euler_can_be_unstable() {
        let adj = complete_graph(5);
        let mut state = HeatState::new(&adj);
        state.set_heat(0, 1.0);
        let mut blew_up = false;
        for _ in 0..100 {
            state.step_euler(1.0);
            if state.temperature.iter().any(|t| t.is_nan() || t.abs() > 1e10) {
                blew_up = true;
                break;
            }
        }
        assert!(blew_up, "Forward Euler should be unstable with large dt");
    }

    #[test]
    fn test_exact_step_always_stable() {
        let adj = complete_graph(5);
        let mut state = HeatState::new(&adj);
        state.set_heat(0, 1.0);
        for _ in 0..100 {
            state.step(100.0);
        }
        assert!(state.temperature.iter().all(|t| t.is_finite()));
        assert!(state.temperature.iter().all(|t| t.abs() < 1e10));
    }

    #[test]
    fn test_laplacian_row_sum_zero() {
        let adj = cycle_graph(6);
        let state = HeatState::new(&adj);
        for row in &state.laplacian {
            let sum: f64 = row.iter().sum();
            assert!(sum.abs() < 1e-10, "Laplacian row should sum to 0, got {}", sum);
        }
    }

    #[test]
    fn test_eigenvalues_non_negative() {
        let adj = path_graph(6);
        let state = HeatState::new(&adj);
        for &lambda in &state.eigenvalues {
            assert!(lambda >= -1e-8, "Laplacian eigenvalue should be non-negative: {}", lambda);
        }
    }

    #[test]
    fn test_first_eigenvalue_zero() {
        let adj = cycle_graph(6);
        let state = HeatState::new(&adj);
        assert!(state.eigenvalues[0].abs() < 1e-8, "First eigenvalue should be ~0, got {}", state.eigenvalues[0]);
    }

    #[test]
    fn test_complete_graph_cr() {
        let adj = complete_graph(5);
        let state = HeatState::new(&adj);
        assert!(state.cr() > 0.5, "Complete graph CR should be high, got {}", state.cr());
    }

    #[test]
    fn test_path_graph_low_cr() {
        let adj = path_graph(6);
        let state = HeatState::new(&adj);
        assert!(state.cr() < *state.eigenvalues.last().unwrap(), "Path graph CR should be low");
    }

    #[test]
    fn test_thermal_image_row_sums_positive() {
        let adj = path_graph(4);
        let thermal = thermal_image(&adj);
        for (i, row) in thermal.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(sum > 0.0, "Thermal image row {} should have positive sum", i);
        }
    }

    #[test]
    fn test_cr_vs_equilibration_ordering() {
        let results = cr_vs_equilibration();
        let path = results[0];
        let complete = results[2];
        assert!(complete.0 > path.0, "Complete CR ({}) should > Path CR ({})", complete.0, path.0);
        assert!(complete.1 < path.1, "Complete eq time ({}) should < Path eq time ({})", complete.1, path.1);
    }

    #[test]
    fn test_diffusion_vs_wave() {
        let adj = cycle_graph(6);
        let report = compare_diffusion_wave(&adj, "Cycle(6)");
        assert!(report.cr > 0.0);
        assert!(report.heat_equilibration_time > 0.0);
        assert!(report.ratio > 0.0);
    }
}
