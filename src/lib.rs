//! # heat-spectral
//!
//! Heat diffusion on graphs with spectral analysis.
//!
//! This crate provides tools for computing heat kernels on graphs via Chebyshev
//! polynomial approximation, spectral filtering for signal denoising, and Fiedler
//! value (algebraic connectivity) computation.
//!
//! ## Core Concepts
//!
//! - **Heat diffusion**: `u(t) = exp(-tL) u(0)` where `L` is the graph Laplacian.
//! - **Equilibration time**: `1 / λ₂` where `λ₂` is the Fiedler value.
//! - **Spectral filtering**: The heat kernel acts as a low-pass filter on graph signals.
//!
//! ## Quick Start
//!
//! ```
//! use heat_spectral::{Graph, HeatKernel, spectral_filter, fiedler};
//!
//! // Build a path graph with 4 nodes
//! let mut g = Graph::new(4);
//! g.add_edge(0, 1, 1.0).unwrap();
//! g.add_edge(1, 2, 1.0).unwrap();
//! g.add_edge(2, 3, 1.0).unwrap();
//!
//! // Compute Fiedler value (algebraic connectivity)
//! let (lambda2, _) = fiedler(&g).unwrap();
//! println!("Fiedler value: {:.4}", lambda2);
//!
//! // Apply heat kernel to a signal
//! let signal = vec![1.0, 0.0, 0.0, 0.0];
//! let mut kernel = HeatKernel::new(g.clone(), 1.0, 10).unwrap();
//! let smoothed = kernel.apply(&signal).unwrap();
//! ```

mod error;
mod graph;
mod laplacian;
mod kernel;
mod diffusion;
mod filter;
mod fiedler;

pub use error::{HeatError, Result};
pub use graph::Graph;
pub use kernel::HeatKernel;
pub use diffusion::{DiffusionResult, DiffusionSimulator};
pub use filter::{spectral_filter, multiscale_filter};
pub use fiedler::{fiedler, algebraic_connectivity};
