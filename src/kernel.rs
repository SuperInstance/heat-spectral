use crate::error::{HeatError, Result};
use crate::graph::Graph;
use crate::laplacian::{mat_vec, max_eigenvalue};

/// Chebyshev-approximated heat kernel on a graph.
///
/// Computes `exp(-τ L) x` in O(k|E|) where `k` is the approximation order.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HeatKernel {
    pub graph: Graph,
    /// Diffusion time parameter τ.
    pub tau: f64,
    /// Chebyshev approximation order.
    pub order: usize,
}

impl HeatKernel {
    /// Create a new heat kernel with the given graph, τ, and approximation order.
    pub fn new(graph: Graph, tau: f64, order: usize) -> Result<Self> {
        if tau < 0.0 {
            return Err(HeatError::NegativeTau { tau });
        }
        if order == 0 {
            return Err(HeatError::InvalidOrder { order });
        }
        Ok(Self { graph, tau, order })
    }

    /// Apply the heat kernel to a signal `x`:  computes `exp(-τ L) x`.
    ///
    /// Uses Chebyshev polynomial approximation of `exp(-τ λ)` on `[0, λ_max]`.
    pub fn apply(&mut self, x: &[f64]) -> Result<Vec<f64>> {
        let n = self.graph.node_count();
        if x.len() != n {
            return Err(HeatError::SignalMismatch { len: x.len(), n });
        }

        // Build Laplacian and find λ_max
        let lap = self.graph.laplacian().to_vec();
        let lambda_max = max_eigenvalue(&lap, 200).max(1e-12);

        // Scale so the eigenvalues lie in [-1, 1]
        // λ_scaled = (2λ / λ_max) - 1,  so λ ∈ [0, λ_max] → [-1, 1]
        let _a = 2.0 * self.tau / lambda_max;

        // Chebyshev approximation of exp(-τ λ) = exp(-a(λ_scaled+1)/2 * λ_max ... )
        // We directly approximate exp(-c * λ) where c = τ, using scaled Laplacian.
        // Shift: L_hat = (2/λ_max) L - I, eigenvalues in [-1,1]
        // exp(-τ L) = exp(-τ λ_max/2 (L_hat + I))

        // Compute Chebyshev coefficients for exp(-τ λ) on [0, λ_max]
        let k = self.order;
        let coeffs = chebyshev_coeffs_exp(self.tau, lambda_max, k);

        // Evaluate the polynomial using Clenshaw recurrence
        // T_0(L_hat) = I, T_1(L_hat) = L_hat
        let scale = 2.0 / lambda_max;

        // L_hat x = scale * L x - x
        let apply_l_hat = |v: &[f64]| -> Vec<f64> {
            let lv = mat_vec(&lap, v);
            v.iter().zip(lv.iter()).map(|(xi, li)| scale * li - xi).collect()
        };

        // Clenshaw: compute sum_k c_k T_k(L_hat) x
        // But we need to adjust: exp(-τ λ) where λ = (λ_max/2)(λ_hat + 1)
        // = exp(-τ λ_max/2) * exp(-τ λ_max/2 λ_hat)
        // The Chebyshev coeffs above are for this shifted function on [-1,1]
        // We just apply the Clenshaw algorithm directly with the L_hat operator.

        let mut bk_plus1 = vec![0.0; n];
        let mut bk = vec![0.0; n];
        let mut bk_minus1 = vec![0.0; n];

        for j in (1..k).rev() {
            std::mem::swap(&mut bk_minus1, &mut bk);
            std::mem::swap(&mut bk, &mut bk_plus1);
            let l_hat_bk = apply_l_hat(&bk);
            let c = if j < coeffs.len() { coeffs[j] } else { 0.0 };
            for i in 0..n {
                bk_minus1[i] = 2.0 * l_hat_bk[i] - bk_plus1[i] + c * x[i];
            }
        }

        // result = c0 * x + c1 * L_hat x - bk_plus1 (which holds b_2 from the first step... )
        // Actually: result = c0 * x + b_1 where b_1 = c1 * x + L_hat * b_0 - b_(-1)... 
        // With Clenshaw: result = c0/2 * x + b_0 but we already used c[j]*x in loop for j=0..k-1
        // Let me redo this properly.

        // Standard Clenshaw for sum_{j=0}^{k-1} c_j T_j(y) applied to matrix:
        // b_{k} = 0, b_{k-1} = c_{k-1} * x
        // b_j = c_j * x + 2 * L_hat * b_{j+1} - b_{j+2}
        // result = c_0 * x + L_hat * b_1 - b_2

        // Reset and do it properly
        let mut b_next = vec![0.0; n]; // b_{k}
        let mut b_cur: Vec<f64>;       // b_{k-1}

        // Start from j = k-1 down to 0
        b_cur = vec![0.0; n];
        let mut b_prev: Vec<f64> = vec![0.0; n];

        for j in (0..k).rev() {
            let tmp = b_next.clone();
            b_next = b_cur.clone();
            b_cur = b_prev.clone();
            b_prev = tmp;

            let c = if j < coeffs.len() { coeffs[j] } else { 0.0 };
            let l_hat_bnext = apply_l_hat(&b_next);
            for i in 0..n {
                b_cur[i] = c * x[i] + 2.0 * l_hat_bnext[i] - b_prev[i];
            }
        }

        // result = c_0 * x + L_hat * b_1 - b_2
        // After loop: b_cur = b_0, b_next = b_1, b_prev = b_2
        // result = c_0 * x + L_hat * b_next - b_prev ... but the standard formula is:
        // result = (c_0) * x + L_hat * b_next - b_prev ... hmm, let me think again.

        // Actually the Clenshaw for scalar is:
        // p = c_0 + y*b_1 - b_2   where b_j = c_j + 2*y*b_{j+1} - b_{j+2}
        // For matrix-vector: result = c_0*x + L_hat*b_1 - b_2
        // After the loop with j going from k-1 to 0:
        //   at the end, b_cur holds b_0 result from last iteration
        //   but we need b_1 and b_2...

        // Let me just use a simpler approach: direct polynomial evaluation
        // p(y) = sum c_j T_j(y),  evaluated with recurrence.

        // Simpler: just evaluate the polynomial directly
        let mut result = vec![0.0; n];
        // c_0 * x
        let c0 = if !coeffs.is_empty() { coeffs[0] } else { 0.0 };
        for i in 0..n {
            result[i] = c0 * x[i];
        }

        if k > 1 {
            let c1 = if coeffs.len() > 1 { coeffs[1] } else { 0.0 };
            let l_hat_x = apply_l_hat(x);
            for i in 0..n {
                result[i] += c1 * l_hat_x[i];
            }
        }

        // For j >= 2: T_j(y) = 2*y*T_{j-1}(y) - T_{j-2}(y)
        let mut t_prev = x.to_vec(); // T_0
        let mut t_cur = if k > 1 { apply_l_hat(x) } else { vec![0.0; n] }; // T_1

        for j in 2..k {
            let t_next = {
                let l_hat_tcur = apply_l_hat(&t_cur);
                let mut tn = vec![0.0; n];
                for i in 0..n {
                    tn[i] = 2.0 * l_hat_tcur[i] - t_prev[i];
                }
                tn
            };
            let cj = if j < coeffs.len() { coeffs[j] } else { 0.0 };
            for i in 0..n {
                result[i] += cj * t_next[i];
            }
            t_prev = t_cur;
            t_cur = t_next;
        }

        Ok(result)
    }
}

/// Compute Chebyshev coefficients for `exp(-τ λ)` where `λ ∈ [0, λ_max]`,
/// mapped to `y ∈ [-1, 1]` via `λ = λ_max (y + 1) / 2`.
///
/// Returns coefficients `c_0, ..., c_{k-1}`.
fn chebyshev_coeffs_exp(tau: f64, lambda_max: f64, k: usize) -> Vec<f64> {
    // f(y) = exp(-τ * λ_max * (y + 1) / 2)
    // Chebyshev coefficient: c_j = (2/π) ∫_{-1}^{1} f(y) T_j(y) / sqrt(1-y^2) dy
    // Approximate with discrete cosine transform on Chebyshev nodes.

    let n_samples = (k * 4).max(64);
    let mut f_vals = vec![0.0; n_samples];

    for (i, val) in f_vals.iter_mut().enumerate() {
        // Chebyshev node: y_i = cos(π (2i+1) / (2n))
        let y = ((2 * i + 1) as f64 * std::f64::consts::PI / (2 * n_samples) as f64).cos();
        let lambda = lambda_max * (y + 1.0) / 2.0;
        *val = (-tau * lambda).exp();
    }

    // DCT-I style: c_j = (2/n) * sum_i f_i * cos(π j (2i+1) / (2n))
    let mut coeffs = Vec::with_capacity(k);
    for j in 0..k {
        let mut sum = 0.0;
        for (i, &fv) in f_vals.iter().enumerate() {
            let angle = std::f64::consts::PI * j as f64 * (2 * i + 1) as f64
                / (2 * n_samples) as f64;
            sum += fv * angle.cos();
        }
        let cj = sum * 2.0 / n_samples as f64;
        coeffs.push(if j == 0 { cj / 2.0 } else { cj });
    }

    coeffs
}
