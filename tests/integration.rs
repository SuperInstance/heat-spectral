#[cfg(test)]
mod tests {
    use heat_spectral::*;

    // ── Graph construction ──

    fn path_graph(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 0..n.saturating_sub(1) {
            g.add_edge(i, i + 1, 1.0).unwrap();
        }
        g
    }

    fn complete_graph(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                g.add_edge(i, j, 1.0).unwrap();
            }
        }
        g
    }

    fn cycle_graph(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 0..n {
            g.add_edge(i, (i + 1) % n, 1.0).unwrap();
        }
        g
    }

    fn star_graph(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 1..n {
            g.add_edge(0, i, 1.0).unwrap();
        }
        g
    }

    // ── Test 1: Graph basics ──

    #[test]
    fn test_graph_node_count() {
        let g = Graph::new(5);
        assert_eq!(g.node_count(), 5);
    }

    #[test]
    fn test_graph_edge_count() {
        let g = path_graph(4);
        assert_eq!(g.edge_count(), 3);
    }

    #[test]
    fn test_complete_graph_edges() {
        let g = complete_graph(4);
        assert_eq!(g.edge_count(), 6);
    }

    #[test]
    fn test_graph_degree() {
        let g = complete_graph(3);
        assert_eq!(g.degree(0), 2.0);
        assert_eq!(g.degree(1), 2.0);
        assert_eq!(g.degree(2), 2.0);
    }

    #[test]
    fn test_star_graph_degrees() {
        let g = star_graph(5);
        assert_eq!(g.degree(0), 4.0);
        for i in 1..5 {
            assert_eq!(g.degree(i), 1.0);
        }
    }

    #[test]
    fn test_invalid_edge() {
        let mut g = Graph::new(3);
        assert!(g.add_edge(0, 5, 1.0).is_err());
    }

    #[test]
    fn test_remove_edge() {
        let mut g = path_graph(3);
        assert!(g.remove_edge(0, 1).unwrap());
        assert_eq!(g.edge_count(), 1);
        assert!(!g.remove_edge(0, 1).unwrap());
    }

    // ── Laplacian ──

    #[test]
    fn test_path_graph_laplacian() {
        let mut g = path_graph(3);
        let l = g.laplacian();
        // Path 0-1-2: L = [[1,-1,0],[-1,2,-1],[0,-1,1]]
        assert!((l[0][0] - 1.0).abs() < 1e-10);
        assert!((l[0][1] - (-1.0)).abs() < 1e-10);
        assert!((l[0][2] - 0.0).abs() < 1e-10);
        assert!((l[1][0] - (-1.0)).abs() < 1e-10);
        assert!((l[1][1] - 2.0).abs() < 1e-10);
        assert!((l[1][2] - (-1.0)).abs() < 1e-10);
        assert!((l[2][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_row_sums_zero() {
        let mut g = complete_graph(5);
        let l = g.laplacian();
        for i in 0..5 {
            let row_sum: f64 = l[i].iter().sum();
            assert!(row_sum.abs() < 1e-10, "row {} sums to {}", i, row_sum);
        }
    }

    #[test]
    fn test_laplacian_diagonal_positive() {
        let mut g = path_graph(5);
        let l = g.laplacian();
        for i in 0..5 {
            assert!(l[i][i] > 0.0, "diagonal {} = {}", i, l[i][i]);
        }
    }

    #[test]
    fn test_normalized_laplacian_diagonal() {
        let g = complete_graph(3);
        let l = g.normalized_laplacian();
        // All diagonals should be 1.0
        for i in 0..3 {
            assert!((l[i][i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_laplacian_cache() {
        let mut g = path_graph(3);
        let l1 = g.laplacian().to_vec();
        let l2 = g.laplacian().to_vec();
        assert_eq!(l1, l2, "cache should return same values");
        g.add_edge(0, 2, 1.0).unwrap();
        let l3 = g.laplacian();
        // After adding edge 0-2, L[0][2] should be -1
        assert!((l3[0][2] - (-1.0)).abs() < 1e-10, "cache should be invalidated after edge add");
    }

    #[test]
    fn test_weighted_laplacian() {
        let mut g = Graph::new(2);
        g.add_edge(0, 1, 3.0).unwrap();
        let l = g.laplacian();
        assert!((l[0][0] - 3.0).abs() < 1e-10);
        assert!((l[0][1] - (-3.0)).abs() < 1e-10);
    }

    // ── Heat kernel ──

    #[test]
    fn test_heat_kernel_tau_zero() {
        let g = path_graph(3);
        let mut kernel = HeatKernel::new(g, 0.0, 5).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let result = kernel.apply(&x).unwrap();
        // exp(0) = I, so result should be approximately x
        for i in 0..3 {
            assert!((result[i] - x[i]).abs() < 0.1, "result[{}] = {} vs {}", i, result[i], x[i]);
        }
    }

    #[test]
    fn test_heat_kernel_preserves_constant() {
        let g = complete_graph(4);
        let mut kernel = HeatKernel::new(g, 1.0, 15).unwrap();
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let result = kernel.apply(&x).unwrap();
        // Constant signal is eigenvector of L with eigenvalue 0
        // exp(0) = 1, so result should be constant
        let mean = result.iter().sum::<f64>() / result.len() as f64;
        for v in &result {
            assert!((v - mean).abs() < 0.2, "deviation from mean: {}", (v - mean).abs());
        }
    }

    #[test]
    fn test_heat_kernel_negative_tau_error() {
        let g = Graph::new(2);
        assert!(HeatKernel::new(g, -1.0, 5).is_err());
    }

    #[test]
    fn test_heat_kernel_zero_order_error() {
        let g = Graph::new(2);
        assert!(HeatKernel::new(g, 1.0, 0).is_err());
    }

    #[test]
    fn test_heat_kernel_signal_mismatch() {
        let g = path_graph(3);
        let mut kernel = HeatKernel::new(g, 1.0, 5).unwrap();
        assert!(kernel.apply(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_heat_kernel_smothers_high_freq() {
        let g = cycle_graph(6);
        let mut kernel = HeatKernel::new(g, 5.0, 15).unwrap();
        // Alternating signal is highest frequency
        let x = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let result = kernel.apply(&x).unwrap();
        // Should be close to zero (high frequency is damped)
        let norm: f64 = result.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm < 1.0, "high frequency should be suppressed, norm = {}", norm);
    }

    // ── Diffusion simulator ──

    #[test]
    fn test_diffusion_conservation() {
        let g = path_graph(4);
        let mut sim = DiffusionSimulator::new(g, 0.01, 100);
        let u0 = vec![10.0, 0.0, 0.0, 0.0];
        let result = sim.run(&u0).unwrap();
        // Conservation ratio should be close to 1.0
        assert!(
            (result.conservation_ratio - 1.0).abs() < 0.1,
            "conservation ratio = {}",
            result.conservation_ratio
        );
    }

    #[test]
    fn test_diffusion_temperatures_length() {
        let g = path_graph(3);
        let steps = 50;
        let mut sim = DiffusionSimulator::new(g, 0.01, steps);
        let u0 = vec![1.0, 0.0, 0.0];
        let result = sim.run(&u0).unwrap();
        assert_eq!(result.temperatures.len(), steps + 1);
        assert_eq!(result.temperatures[0].len(), 3);
    }

    #[test]
    fn test_diffusion_equilibrates() {
        let g = path_graph(4);
        let mut sim = DiffusionSimulator::new(g, 0.005, 2000);
        let u0 = vec![10.0, 0.0, 0.0, 0.0];
        let result = sim.run(&u0).unwrap();
        let final_temps = result.temperatures.last().unwrap();
        let mean = final_temps.iter().sum::<f64>() / final_temps.len() as f64;
        for t in final_temps {
            assert!((t - mean).abs() < 1.0, "not equilibrated: t = {}", t);
        }
    }

    // ── Spectral filter ──

    #[test]
    fn test_spectral_filter_basic() {
        let mut g = path_graph(5);
        let signal = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let filtered = spectral_filter(&mut g, &signal, 1.0, 10).unwrap();
        assert_eq!(filtered.len(), 5);
        // Total energy should be roughly conserved
        let input_sum: f64 = signal.iter().sum();
        let output_sum: f64 = filtered.iter().sum();
        assert!((output_sum - input_sum).abs() < 2.0, "sum changed too much");
    }

    #[test]
    fn test_spectral_filter_denoises() {
        let mut g = cycle_graph(10);
        // Low-frequency signal + noise
        let mut signal = vec![0.0; 10];
        for i in 0..10 {
            signal[i] = (i as f64 / 10.0 * std::f64::consts::PI).sin() + 0.3 * ((i * 7) as f64).sin();
        }
        let filtered = spectral_filter(&mut g, &signal, 2.0, 15).unwrap();
        // Filtered should be smoother (less variation between neighbors)
        let mut variation_before = 0.0;
        let mut variation_after = 0.0;
        for i in 0..10 {
            let j = (i + 1) % 10;
            variation_before += (signal[i] - signal[j]).powi(2);
            variation_after += (filtered[i] - filtered[j]).powi(2);
        }
        assert!(variation_after < variation_before, "filter should reduce variation");
    }

    // ── Fiedler value ──

    #[test]
    fn test_fiedler_path_graph() {
        let g = path_graph(4);
        let (lambda2, _) = fiedler(&g).unwrap();
        // Known: λ₂ of path P₄ = 2 - 2cos(π/4) ≈ 0.5858
        let expected = 2.0 - 2.0 * (std::f64::consts::PI / 4.0).cos();
        assert!(
            (lambda2 - expected).abs() < 0.15,
            "λ₂ = {} vs expected {}",
            lambda2,
            expected
        );
    }

    #[test]
    fn test_fiedler_complete_graph() {
        let g = complete_graph(4);
        let (lambda2, _) = fiedler(&g).unwrap();
        // K₄ has λ₂ = n = 4
        assert!(
            (lambda2 - 4.0).abs() < 0.5,
            "λ₂ = {} vs expected 4.0",
            lambda2
        );
    }

    #[test]
    fn test_fiedler_star_graph() {
        let g = star_graph(5);
        let (lambda2, _) = fiedler(&g).unwrap();
        // Star S₅ has λ₂ = 1.0
        assert!(
            (lambda2 - 1.0).abs() < 0.3,
            "λ₂ = {} vs expected 1.0",
            lambda2
        );
    }

    #[test]
    fn test_algebraic_connectivity() {
        let g = complete_graph(3);
        let ac = algebraic_connectivity(&g).unwrap();
        assert!(ac > 0.0, "algebraic connectivity should be positive for connected graph");
    }

    #[test]
    fn test_fiedler_too_few_nodes() {
        let g = Graph::new(1);
        assert!(fiedler(&g).is_err());
    }

    #[test]
    fn test_fiedler_eigenvalue_gap() {
        // Complete graph should have larger λ₂ than path graph of same size
        let g_complete = complete_graph(5);
        let g_path = path_graph(5);
        let (lc, _) = fiedler(&g_complete).unwrap();
        let (lp, _) = fiedler(&g_path).unwrap();
        assert!(lc > lp, "complete λ₂ ({}) should > path λ₂ ({})", lc, lp);
    }

    // ── Multiscale filter ──

    #[test]
    fn test_multiscale_filter() {
        let mut g = cycle_graph(6);
        let signal = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let result = multiscale_filter(&mut g, &signal, &[0.5, 1.0, 2.0], 10).unwrap();
        assert_eq!(result.len(), 6);
    }

    // ── Edge cases ──

    #[test]
    fn test_empty_graph() {
        let g = Graph::new(0);
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_single_node() {
        let g = Graph::new(1);
        assert_eq!(g.degree(0), 0.0);
    }

    #[test]
    fn test_diffusion_signal_mismatch() {
        let g = path_graph(3);
        let mut sim = DiffusionSimulator::new(g, 0.01, 10);
        assert!(sim.run(&[1.0]).is_err());
    }
}
