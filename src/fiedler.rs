use crate::error::{HeatError, Result};
use crate::graph::Graph;
use crate::laplacian::{build_laplacian, mat_vec};

/// Compute the Fiedler value (second smallest eigenvalue of the graph Laplacian, λ₂)
/// using deflated power iteration.
///
/// Returns `(lambda2, fiedler_vector)` where the Fiedler vector is the corresponding
/// eigenvector (normalized to unit length).
pub fn fiedler(graph: &Graph) -> Result<(f64, Vec<f64>)> {
    let n = graph.node_count();
    if n < 2 {
        return Err(HeatError::TooFewNodes);
    }

    let lap = build_laplacian(graph);

    // First, find the smallest eigenvalue (should be ~0 for connected graph) and its eigenvector
    let (lambda1, v1) = power_iteration_smallest(&lap, 300);

    // Check if graph is connected: smallest eigenvalue should be ~0
    if lambda1 > 0.1 {
        // Disconnected graph: λ₁ > 0 but we can still return it as λ₂ = 0
        return Ok((0.0, v1));
    }

    // Deflate: find the next smallest eigenvalue by projecting out v1
    let (lambda2, v2) = power_iteration_smallest_deflated(&lap, &v1, 300);

    Ok((lambda2, v2))
}

/// Algebraic connectivity: same as the Fiedler value.
pub fn algebraic_connectivity(graph: &Graph) -> Result<f64> {
    let (lambda2, _) = fiedler(graph)?;
    Ok(lambda2)
}

/// Power iteration to find the smallest eigenvalue of `mat`.
/// Uses inverse iteration: solve (mat - σI)x = b with σ near the smallest eigenvalue.
/// For the Laplacian, we know the smallest eigenvalue is 0, so we shift slightly.
fn power_iteration_smallest(mat: &[Vec<f64>], iters: usize) -> (f64, Vec<f64>) {
    let n = mat.len();
    // Use inverse iteration with shift near 0
    // (L - σI)⁻¹ x where σ = -epsilon (so we find eigval closest to -epsilon ≈ 0)
    let sigma = -1e-8;
    let mut v: Vec<f64> = (0..n).map(|i| i as f64 + 1.0 ).collect();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v { *x /= norm; }

    for _ in 0..iters {
        // Solve (L - σI) y = v using simple Gauss-Seidel
        let y = solve_shifted(mat, sigma, &v, 50);
        let norm = y.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            v = y.iter().map(|x| x / norm).collect();
        }
    }

    // Compute eigenvalue: λ = v^T L v
    let lv = mat_vec(mat, &v);
    let lambda: f64 = v.iter().zip(lv.iter()).map(|(a, b)| a * b).sum();

    (lambda, v)
}

/// Inverse iteration with deflation: find the smallest eigenvalue orthogonal to `v1`.
fn power_iteration_smallest_deflated(
    mat: &[Vec<f64>],
    v1: &[f64],
    iters: usize,
) -> (f64, Vec<f64>) {
    let n = mat.len();
    let sigma = -1e-8;

    // Start with a random vector orthogonal to v1
    let mut v: Vec<f64> = (0..n).map(|i| i as f64 + 1.0 ).collect();
    // Project out v1
    let dot: f64 = v.iter().zip(v1.iter()).map(|(a, b)| a * b).sum();
    for i in 0..n { v[i] -= dot * v1[i]; }
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-15 {
        // Degenerate case
        return (0.0, vec![0.0; n]);
    }
    for x in &mut v { *x /= norm; }

    for _ in 0..iters {
        let mut y = solve_shifted(mat, sigma, &v, 50);
        // Project out v1
        let dot: f64 = y.iter().zip(v1.iter()).map(|(a, b)| a * b).sum();
        for i in 0..n { y[i] -= dot * v1[i]; }
        let norm = y.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            v = y.iter().map(|x| x / norm).collect();
        }
    }

    let lv = mat_vec(mat, &v);
    let lambda: f64 = v.iter().zip(lv.iter()).map(|(a, b)| a * b).sum();

    (lambda, v)
}

/// Solve (L - σI) x = b using Gauss-Seidel iteration.
fn solve_shifted(lap: &[Vec<f64>], sigma: f64, b: &[f64], iters: usize) -> Vec<f64> {
    let n = lap.len();
    let mut x = b.to_vec();

    for _ in 0..iters {
        for i in 0..n {
            let diag = lap[i][i] - sigma;
            if diag.abs() < 1e-15 {
                continue;
            }
            let mut sum = b[i];
            for j in 0..n {
                if j != i {
                    sum -= lap[i][j] * x[j];
                }
            }
            x[i] = sum / diag;
        }
    }

    x
}
