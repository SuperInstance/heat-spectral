use std::f64::consts::PI;

// ============================================================
// Module 1: HeatDiffusion
// ============================================================

pub struct HeatState {
    pub temperature: Vec<f64>,
    laplacian: Vec<Vec<f64>>,
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Vec<Vec<f64>>,
    initial_heat: f64,
}

impl HeatState {
    pub fn new(adj: &[Vec<f64>]) -> HeatState {
        let n = adj.len();
        // Build Laplacian: L = D - A
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

    pub fn set_heat(&mut self, node: usize, temp: f64) {
        self.temperature = vec![0.0; self.eigenvalues.len()];
        self.temperature[node] = temp;
        self.initial_heat = temp;
    }

    pub fn set_pattern(&mut self, temps: &[f64]) {
        self.temperature = temps.to_vec();
        self.initial_heat = temps.iter().sum();
    }

    /// Exact step: project onto eigenvectors, multiply by exp(-λ_i * dt)
    pub fn step(&mut self, dt: f64) {
        let n = self.eigenvalues.len();
        let mut new_temp = vec![0.0; n];

        for k in 0..n {
            let lambda = self.eigenvalues[k];
            // Eigenvectors stored as columns: eigenvectors[k] is the k-th eigenvector
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

    /// Forward Euler step (can be unstable for large dt)
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

    pub fn temperature(&self) -> &[f64] {
        &self.temperature
    }

    pub fn total_heat(&self) -> f64 {
        self.temperature.iter().sum()
    }

    pub fn variance(&self) -> f64 {
        let n = self.temperature.len() as f64;
        let mean = self.temperature.iter().sum::<f64>() / n;
        self.temperature.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n
    }

    pub fn equilibration_time(&self) -> f64 {
        for &lambda in &self.eigenvalues {
            if lambda > 1e-10 {
                return 1.0 / lambda;
            }
        }
        f64::INFINITY
    }

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

pub struct DiffusionReport {
    pub initial_variance: f64,
    pub variance_at_step: Vec<f64>,
    pub equilibration_time: f64,
    pub predicted_eq_time: f64,
    pub heat_conservation_error: f64,
    pub cr: f64,
}

// ============================================================
// Module 2: DiffusionExperiments
// ============================================================

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

    // Use variance threshold at 1/e² of initial (closer to true equilibration)
    let threshold = initial_variance * 0.01;
    let eq_step = variance_at_step.iter().position(|&v| v < threshold).unwrap_or(variance_at_step.len());
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

// ============================================================
// Module 3: DiffusionVsWave
// ============================================================

pub struct ComparisonReport {
    pub graph_name: String,
    pub cr: f64,
    pub heat_equilibration_time: f64,
    pub wave_coherence_halflife: usize,
    pub ratio: f64,
}

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

    // Wave: ∂²u/∂t² = -Lu (leapfrog)
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
    let cov: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - mean_a) * (y - mean_b)).sum();
    let var_a: f64 = a.iter().map(|x| (x - mean_a).powi(2)).sum();
    let var_b: f64 = b.iter().map(|x| (x - mean_b).powi(2)).sum();
    if var_a < 1e-15 || var_b < 1e-15 {
        return 0.0;
    }
    cov / (var_a.sqrt() * var_b.sqrt())
}

// ============================================================
// Graph generators
// ============================================================

pub fn path_graph(n: usize) -> Vec<Vec<f64>> {
    let mut adj = vec![vec![0.0; n]; n];
    for i in 0..n - 1 {
        adj[i][i + 1] = 1.0;
        adj[i + 1][i] = 1.0;
    }
    adj
}

pub fn cycle_graph(n: usize) -> Vec<Vec<f64>> {
    let mut adj = path_graph(n);
    adj[0][n - 1] = 1.0;
    adj[n - 1][0] = 1.0;
    adj
}

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

pub fn star_graph(n: usize) -> Vec<Vec<f64>> {
    let mut adj = vec![vec![0.0; n]; n];
    for i in 1..n {
        adj[0][i] = 1.0;
        adj[i][0] = 1.0;
    }
    adj
}

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

// ============================================================
// Jacobi eigenvalue algorithm for symmetric matrices
// ============================================================

fn jacobi_eigen(mat: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = mat.len();
    let mut a = mat.to_vec();
    // V accumulates rotations (columns are eigenvectors)
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    let max_sweeps = 100;
    let tol = 1e-12;

    for _ in 0..max_sweeps {
        // Find largest off-diagonal element
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

        // Compute rotation angle
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

        // Apply rotation: A' = J^T A J
        // Only rows/cols p,q change
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

        // Update eigenvectors: V' = V J
        let mut new_v = v.clone();
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            new_v[i][p] = c * vip + s * viq;
            new_v[i][q] = -s * vip + c * viq;
        }
        v = new_v;
    }

    // Extract eigenvalues and sort
    let mut indexed: Vec<(usize, f64)> = (0..n).map(|i| (i, a[i][i])).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let eigenvalues: Vec<f64> = indexed.iter().map(|(_, v)| *v).collect();
    // v columns -> reorder; v[i][j] = component i of eigenvector j
    let eigenvectors: Vec<Vec<f64>> = indexed
        .iter()
        .map(|(j, _)| (0..n).map(|i| v[i][*j]).collect())
        .collect();

    (eigenvalues, eigenvectors)
}

// ============================================================
// Main
// ============================================================

fn main() {
    println!("=== Heat Spectral: Heat Diffusion on Graphs ===\n");

    println!("--- Experiment 1: Equilibration Time ---");
    let adj = cycle_graph(8);
    let report = verify_equilibration_time(&adj);
    println!("Graph: Cycle(8)");
    println!("  CR: {:.4}", report.cr);
    println!("  Predicted eq time (1/λ₂): {:.4}", report.predicted_eq_time);
    println!("  Measured eq time: {:.4}", report.equilibration_time);
    println!("  Heat conservation error: {:.2e}", report.heat_conservation_error);
    println!();

    println!("--- Experiment 2: CR vs Equilibration Speed ---");
    let results = cr_vs_equilibration();
    println!("{:<15} {:>10} {:>20}", "Graph", "CR", "1/λ₂ (eq time)");
    let names = ["path-6", "cycle-6", "complete-6", "star-6", "barbell-4"];
    for (i, (cr, eq)) in results.iter().enumerate() {
        println!("{:<15} {:>10.4} {:>20.4}", names[i], cr, eq);
    }
    println!();

    println!("--- Experiment 3: Spectral Filtering ---");
    let n = 8;
    let adj = cycle_graph(n);
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / n as f64).sin() + 0.3 * (6.0 * PI * i as f64 / n as f64).sin())
        .collect();
    let short = spectral_filtering(&signal, &adj, 0.05);
    let long = spectral_filtering(&signal, &adj, 2.0);
    println!(
        "Signal:       {:?}",
        signal.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>()
    );
    println!(
        "Short (0.05): {:?}",
        short.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>()
    );
    println!(
        "Long  (2.00): {:?}",
        long.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>()
    );
    println!();

    println!("--- Experiment 4: Heat Anomaly Detection ---");
    let adj = cycle_graph(8);
    let normal_dist = heat_anomaly_detection(&adj, 0);
    println!("Cycle(8): node 0 deviation: {:.6}", normal_dist);
    let adj2 = barbell_graph(4);
    let d1 = heat_anomaly_detection(&adj2, 0);
    let d2 = heat_anomaly_detection(&adj2, 3);
    println!("Barbell: node 0 deviation: {:.6}", d1);
    println!("Barbell: bridge node 3 deviation: {:.6}", d2);
    println!();

    println!("--- Experiment 5: Thermal Image (Path-4) ---");
    let adj = path_graph(4);
    let thermal = thermal_image(&adj);
    for (i, row) in thermal.iter().enumerate() {
        print!("Source {}: ", i);
        for &t in row {
            print!("{:.4} ", t);
        }
        println!();
    }
    println!();

    println!("--- Diffusion vs Wave Comparison ---");
    for (adj, name) in [
        (path_graph(6), "Path(6)"),
        (cycle_graph(6), "Cycle(6)"),
        (complete_graph(6), "Complete(6)"),
    ] {
        let cmp = compare_diffusion_wave(&adj, name);
        println!(
            "{:<12} CR={:.4} heat_eq={:.4} wave_half={} ratio={:.4}",
            cmp.graph_name, cmp.cr, cmp.heat_equilibration_time, cmp.wave_coherence_halflife, cmp.ratio
        );
    }
}

// ============================================================
// Tests
// ============================================================

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
    fn test_heat_anomaly_barbell() {
        let adj = barbell_graph(4);
        let bridge_dist = heat_anomaly_detection(&adj, 3);
        let interior_dist = heat_anomaly_detection(&adj, 0);
        assert!(
            bridge_dist >= 0.0 && interior_dist >= 0.0,
            "Anomaly detection should produce non-negative values"
        );
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
            assert!(
                sum.abs() < 1e-10,
                "Laplacian row should sum to 0, got {}",
                sum
            );
        }
    }

    #[test]
    fn test_eigenvalues_non_negative() {
        let adj = path_graph(6);
        let state = HeatState::new(&adj);
        for &lambda in &state.eigenvalues {
            assert!(
                lambda >= -1e-8,
                "Laplacian eigenvalue should be non-negative: {}",
                lambda
            );
        }
    }

    #[test]
    fn test_first_eigenvalue_zero() {
        let adj = cycle_graph(6);
        let state = HeatState::new(&adj);
        assert!(
            state.eigenvalues[0].abs() < 1e-8,
            "First eigenvalue should be ~0, got {}",
            state.eigenvalues[0]
        );
    }

    #[test]
    fn test_complete_graph_cr() {
        let adj = complete_graph(5);
        let state = HeatState::new(&adj);
        assert!(
            state.cr() > 0.5,
            "Complete graph CR should be high, got {}",
            state.cr()
        );
    }

    #[test]
    fn test_path_graph_low_cr() {
        let adj = path_graph(6);
        let state = HeatState::new(&adj);
        assert!(
            state.cr() < *state.eigenvalues.last().unwrap(),
            "Path graph CR should be low"
        );
    }

    #[test]
    fn test_thermal_image_row_sums_positive() {
        let adj = path_graph(4);
        let thermal = thermal_image(&adj);
        for (i, row) in thermal.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                sum > 0.0,
                "Thermal image row {} should have positive sum",
                i
            );
        }
    }

    #[test]
    fn test_cr_vs_equilibration_ordering() {
        let results = cr_vs_equilibration();
        let path = results[0];
        let complete = results[2];
        assert!(
            complete.0 > path.0,
            "Complete CR ({}) should > Path CR ({})",
            complete.0,
            path.0
        );
        assert!(
            complete.1 < path.1,
            "Complete eq time ({}) should < Path eq time ({})",
            complete.1,
            path.1
        );
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
