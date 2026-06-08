use crate::error::{HeatError, Result};

/// Undirected weighted graph stored as an adjacency list.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Graph {
    adj: Vec<Vec<(usize, f64)>>,
    n: usize,
    #[serde(skip)]
    laplacian_cached: Option<Vec<Vec<f64>>>,
}

impl Graph {
    /// Create an empty graph with `n` nodes and no edges.
    pub fn new(n: usize) -> Self {
        Self {
            adj: vec![Vec::new(); n],
            n,
            laplacian_cached: None,
        }
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.n
    }

    /// Number of edges (counting each undirected edge once).
    pub fn edge_count(&self) -> usize {
        let total: usize = self.adj.iter().map(|v| v.len()).sum();
        total / 2
    }

    /// Degree (sum of weights) of node `i`.
    pub fn degree(&self, i: usize) -> f64 {
        self.adj[i].iter().map(|(_, w)| w).sum()
    }

    /// Add an undirected edge `(i, j)` with weight `w`.  Invalidates the Laplacian cache.
    pub fn add_edge(&mut self, i: usize, j: usize, w: f64) -> Result<()> {
        if i >= self.n || j >= self.n {
            return Err(HeatError::InvalidNode {
                index: i.max(j),
                n: self.n,
            });
        }
        self.adj[i].push((j, w));
        self.adj[j].push((i, w));
        self.laplacian_cached = None;
        Ok(())
    }

    /// Remove the undirected edge `(i, j)`.  Returns `true` if an edge existed.
    pub fn remove_edge(&mut self, i: usize, j: usize) -> Result<bool> {
        if i >= self.n || j >= self.n {
            return Err(HeatError::InvalidNode {
                index: i.max(j),
                n: self.n,
            });
        }
        let had_i = self.adj[i].iter().position(|(v, _)| *v == j).map(|p| self.adj[i].remove(p).1);
        let had_j = self.adj[j].iter().position(|(v, _)| *v == i).map(|p| self.adj[j].remove(p).1);
        let removed = had_i.is_some() || had_j.is_some();
        if removed {
            self.laplacian_cached = None;
        }
        Ok(removed)
    }

    /// Adjacency list reference for node `i`.
    pub fn neighbors(&self, i: usize) -> &[(usize, f64)] {
        &self.adj[i]
    }

    /// Compute the **unnormalized** Laplacian `L = D - A` (cached).
    pub fn laplacian(&mut self) -> &[Vec<f64>] {
        if self.laplacian_cached.is_none() {
            let n = self.n;
            let mut l = vec![vec![0.0; n]; n];
            for (i, row) in l.iter_mut().enumerate() {
                let deg = self.degree(i);
                row[i] = deg;
                for &(j, w) in &self.adj[i] {
                    row[j] -= w;
                }
            }
            self.laplacian_cached = Some(l);
        }
        self.laplacian_cached.as_ref().unwrap()
    }

    /// Compute the **symmetric normalized** Laplacian `L_sym = I - D^{-1/2} A D^{-1/2}`.
    pub fn normalized_laplacian(&self) -> Vec<Vec<f64>> {
        let n = self.n;
        let mut l = vec![vec![0.0; n]; n];
        let inv_sqrt_deg: Vec<f64> = (0..n)
            .map(|i| {
                let d = self.degree(i);
                if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }
            })
            .collect();
        for i in 0..n {
            l[i][i] = 1.0;
            for &(j, w) in &self.adj[i] {
                l[i][j] -= inv_sqrt_deg[i] * w * inv_sqrt_deg[j];
            }
        }
        l
    }

    /// Invalidate the cached Laplacian (called automatically on edge changes).
    pub fn invalidate_cache(&mut self) {
        self.laplacian_cached = None;
    }
}
