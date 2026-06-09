# heat-spectral

**Exact spectral solution of the discrete heat equation on arbitrary graphs — zero dependencies, unconditional stability, machine-precision energy conservation.**

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## The Problem

You have a graph — a social network, a road map, a neural connectivity matrix, a mesh for finite elements — and you need to understand how information, energy, or influence *diffuses* through it. The discrete heat equation ∂u/∂t = −Lu (where L is the graph Laplacian) is the canonical model. But numerical integration (Euler, RK4) introduces drift, has stability limits on the time step, and accumulates error. For a process that's fundamentally linear and analytically solvable, numerical approximation is the wrong tool.

## The Key Insight

The heat equation on a graph is a **linear ODE with constant coefficients**. The eigenvectors of the Laplacian form a basis that diagonalizes the problem. If you decompose the temperature vector u(0) = Σ cᵢvᵢ onto the eigenvector basis, the exact solution at any future time is simply:

**u(t) = Σ cᵢ · exp(−λᵢ · t) · vᵢ**

Each eigenmode decays independently at a rate determined by its eigenvalue. High-frequency modes (large λ) die fast; the lowest non-zero mode (the Fiedler vector, λ₂) is the slowest to equilibrate. This isn't an approximation — it's the *exact* answer. No drift, no stability condition, no error accumulation. The conservation ratio CR = λ₂/λₙ captures the whole picture: how quickly the graph mixes, in a single number.

## Architecture

```
┌─────────────────────┐
│   Adjacency Matrix   │    Weighted graph structure (user-supplied)
│   A[i][j] = w_ij     │
└──────────┬──────────┘
           │  L = D − A (degree matrix minus adjacency)
┌──────────▼──────────┐
│   Graph Laplacian    │    Symmetric, positive semi-definite
│   Row sums → 0       │    λ₁ ≈ 0, λ₂…λₙ ≥ 0
└──────────┬──────────┘
           │  Jacobi eigenvalue rotation (O(n³), zero deps)
┌──────────▼──────────┐
│   Eigendecomposition │    λ₁ = 0 (constant mode), λ₂ = algebraic connectivity
│   L = V Λ Vᵀ         │    V = orthonormal eigenvectors
└──────────┬──────────┘
           │  Project: cᵢ = ⟨u, vᵢ⟩
           │  Evolve:  cᵢ → cᵢ · exp(−λᵢ · Δt)
           │  Reconstruct: u = Σ cᵢ vᵢ
┌──────────▼──────────┐
│   Exact Time Step    │    Unconditionally stable, no truncation error
│   O(n²) per step    │    Total heat conserved to machine precision
└──────────┬──────────┘
           │
     ┌─────┼──────────────┬────────────────┬───────────────┐
     │     │              │                │               │
┌────▼──┐ ┌▼────────┐ ┌───▼────────┐ ┌────▼────────┐ ┌────▼─────────┐
│Filter │ │Anomaly  │ │  Thermal   │ │  Equilib.   │ │ Diffusion vs │
│(blur) │ │ detect  │ │  imaging   │ │  experiments│ │    Wave      │
└───────┘ └─────────┘ └────────────┘ └─────────────┘ └──────────────┘
```

## Quick Start

```rust
use heat_spectral::{HeatState, cycle_graph};

// Create an 8-node cycle graph and initialize diffusion
let adj = cycle_graph(8);
let mut sim = HeatState::new(&adj);

// Place a heat pulse at node 0
sim.set_heat(0, 1.0);

// Advance time with exact spectral steps — any dt works
for _ in 0..100 {
    sim.step(0.1);
}

// Observe conservation and equilibration
println!("Total heat:  {:.10}", sim.total_heat());  // 1.0000000000
println!("Variance:    {:.6}", sim.variance());     // → 0 (equilibrium)
println!("CR (λ₂/λₙ): {:.4}", sim.cr());           // mixing quality
```

## Tutorial

### Building Graphs

The crate provides generators for common graph families:

```rust
use heat_spectral::*;

let path     = path_graph(10);      // Chain: 0-1-2-...-9  (slowest mixing)
let cycle    = cycle_graph(10);     // Ring: closes the loop
let complete = complete_graph(10);  // Every node ↔ every node (fastest mixing)
let star     = star_graph(10);      // Hub-and-spoke
let barbell  = barbell_graph(5);    // Two 5-cliques joined by bridge (bottleneck)
```

### Reading the Spectrum

```rust
let state = HeatState::new(&cycle_graph(8));

// Eigenvalues reveal connectivity
println!("λ₁ = {:.6} (should be ≈0)", state.eigenvalues[0]);
println!("λ₂ = {:.6} (algebraic connectivity)", state.eigenvalues[1]);
println!("λₙ = {:.6} (spectral radius)", state.eigenvalues.last().unwrap());

// The conservation ratio: CR = λ₂ / λₙ
println!("CR = {:.4}", state.cr());
// CR > 0.7 → fast mixing (expander-like)
// CR < 0.1 → severe bottleneck
```

### Spectral Filtering (Graph Gaussian Blur)

```rust
// A noisy signal on a 12-node cycle
let signal: Vec<f64> = (0..12).map(|i| {
    (2.0 * PI * i as f64 / 12.0).sin()  // clean sinusoid
    + if i % 3 == 0 { 0.5 } else { 0.0 } // impulse noise
}).collect();

// Short diffusion: preserves structure, removes noise
let smoothed = spectral_filtering(&signal, &cycle_graph(12), 0.1);

// Long diffusion: approaches uniform (the DC component)
let uniform = spectral_filtering(&signal, &cycle_graph(12), 5.0);
```

### Finding Anomalous Nodes

```rust
// Barbell graph: two cliques joined by a single bridge
let adj = barbell_graph(5);

// Bridge nodes have atypical thermal signatures
let bridge_deviation = heat_anomaly_detection(&adj, 4);   // bridge node
let interior_deviation = heat_anomaly_detection(&adj, 0); // interior node
// bridge_deviation > interior_deviation
```

### Heat Diffusion vs Wave Propagation

```rust
let report = compare_diffusion_wave(&path_graph(8), "Path(8)");
// Heat equilibrates (dissipative) — variance → 0
// Waves oscillate (conservative) — coherence decays slowly
// Ratio reveals topology-dependent interplay
```

Run the full tutorial: `cargo run --example tutorial`

## API Reference

| Type / Function | Module | Description |
|---|---|---|
| `HeatState::new(&adj)` | Core | Create state from adjacency matrix (computes eigendecomposition) |
| `HeatState::set_heat(node, temp)` | Core | Set point heat source |
| `HeatState::set_pattern(&temps)` | Core | Set arbitrary temperature distribution |
| `HeatState::step(dt)` | Core | Exact spectral time-step (unconditionally stable) |
| `HeatState::step_euler(dt)` | Core | Forward Euler step (unstable for large dt — for comparison) |
| `HeatState::total_heat()` | Core | Sum of temperatures (conserved by spectral step) |
| `HeatState::variance()` | Core | Temperature variance (monotonically decreasing) |
| `HeatState::cr()` | Core | Conservation ratio CR = λ₂/λₙ |
| `HeatState::equilibration_time()` | Core | Predicted eq. time = 1/λ₂ |
| `verify_equilibration_time(&adj)` | Experiments | Full equilibration experiment with report |
| `spectral_filtering(&signal, &adj, time)` | Experiments | Low-pass filter via heat diffusion |
| `heat_anomaly_detection(&adj, node)` | Experiments | Thermal fingerprint anomaly score |
| `thermal_image(&adj)` | Experiments | n×n matrix of source→destination heat flows |
| `compare_diffusion_wave(&adj, name)` | Experiments | Heat vs wave comparison report |
| `path_graph(n)` | Generators | Path (chain) topology |
| `cycle_graph(n)` | Generators | Cycle (ring) topology |
| `complete_graph(n)` | Generators | Complete (clique) topology |
| `star_graph(n)` | Generators | Star (hub-spoke) topology |
| `barbell_graph(m)` | Generators | Two m-cliques + bridge edge |
| `DiffusionReport` | Reports | Equilibration experiment results |
| `ComparisonReport` | Reports | Diffusion vs wave comparison results |

## Why Eigenvalues Capture Diffusion

The graph Laplacian L is a positive semi-definite matrix. Its eigenvalue decomposition L = VΛVᵀ is the spectral analog of the Fourier transform:

- **λ₁ = 0** corresponds to the constant vector — uniform temperature doesn't change.
- **λ₂** (the Fiedler value / algebraic connectivity) is the smallest non-zero eigenvalue. It controls the *slowest* decaying mode. The inverse 1/λ₂ is the equilibration time constant.
- **λₙ** (the spectral radius) controls the *fastest* decaying mode. High-frequency spatial patterns die at rate λₙ.
- **CR = λ₂/λₙ** measures how "spread out" the spectrum is. High CR → all modes decay together → fast mixing. Low CR → some modes persist much longer → bottlenecks.

This is why the Jacobi eigendecomposition is the heart of this crate: once you have the eigenbasis, the entire future evolution of the system is determined by the simple formula u(t) = Σ cᵢ·exp(−λᵢt)·vᵢ.

## Zero Dependencies

This crate uses **no external crates**. The Jacobi eigenvalue algorithm is implemented from scratch, requiring only `std::f64::consts::PI`. No `ndarray`, no `nalgebra`, no BLAS. Pure Rust, portable, auditable.

## Installation

```toml
[dependencies]
heat-spectral = { git = "https://github.com/SuperInstance/heat-spectral" }
```

## Running

```bash
cargo run                      # Built-in experiments
cargo run --example tutorial   # Guided walkthrough
cargo run --example advanced   # Real-world applications
cargo test                     # Full test suite
```

## Ecosystem Role

**heat-spectral** is the parabolic PDE layer in the SuperInstance spectral analysis framework:

- **heat-spectral** — Heat diffusion (parabolic PDE) on graphs *(this crate)*
- **[wave-conservation](https://github.com/SuperInstance/wave-conservation)** — Wave propagation (hyperbolic PDE) on graphs
- **[spectral-fingerprint](https://github.com/SuperInstance/spectral-fingerprint)** — Spectral code similarity via eigenvalue fingerprinting
- **[graph-neural](https://github.com/SuperInstance/graph-neural)** — Graph neural network spectral primitives

## License

MIT
