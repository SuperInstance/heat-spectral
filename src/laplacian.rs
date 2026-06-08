use crate::graph::Graph;

/// Build the unnormalized Laplacian `L = D - A` as a dense matrix.
pub fn build_laplacian(graph: &Graph) -> Vec<Vec<f64>> {
    let n = graph.node_count();
    let mut l = vec![vec![0.0; n]; n];
    for (i, row) in l.iter_mut().enumerate() {
        let deg = graph.degree(i);
        row[i] = deg;
        for &(j, w) in graph.neighbors(i) {
            row[j] -= w;
        }
    }
    l
}

/// Build the symmetric normalized Laplacian `L_sym = I - D^{-1/2} A D^{-1/2}`.
#[allow(dead_code)]
pub fn build_normalized_laplacian(graph: &Graph) -> Vec<Vec<f64>> {
    graph.normalized_laplacian()
}

/// Sparse matrix-vector multiply `y = M x` for a dense matrix.
pub fn mat_vec(mat: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    mat.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Estimate the largest eigenvalue via power iteration.
pub fn max_eigenvalue(mat: &[Vec<f64>], iters: usize) -> f64 {
    let n = mat.len();
    if n == 0 {
        return 0.0;
    }
    let mut v: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v { *x /= norm; }

    let mut lambda = 0.0;
    for _ in 0..iters {
        let mv = mat_vec(mat, &v);
        lambda = v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
        let norm = mv.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            v = mv.iter().map(|x| x / norm).collect();
        }
    }
    lambda
}
