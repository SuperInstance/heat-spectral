# heat-spectral

**Heat diffusion on graphs — exact spectral solution via Jacobi eigendecomposition, zero external dependencies.**

Simulates the discrete heat equation ∂u/∂t = -L·u on arbitrary graph structures. Uses exact eigendecomposition (Jacobi rotation) for time-stepping: project onto eigenvectors, multiply by exp(-λᵢ·dt), reconstruct. No iterative solvers, no approximation.

## What This Gives You

- **Exact heat diffusion** — no numerical drift, energy preserved by construction
- **Jacobi eigendecomposition** — all eigenvalues and eigenvectors computed from scratch
- **Multiple initial conditions** — single-node heat pulse, arbitrary temperature patterns
- **Conservation tracking** — total heat is conserved across diffusion steps
- **Graph spectrum visualization** — eigenvalue decay reveals graph connectivity
- **Zero dependencies** — pure Rust, no linear algebra crates

## Quick Start

```rust
use heat_spectral::HeatState;

// Build adjacency matrix for a 10-node path graph
let adj = path_graph(10);

let mut sim = HeatState::new(&adj);
sim.set_heat(0, 1.0);  // heat pulse at node 0

for _ in 0..100 {
    sim.step(0.1);  // exact step: eigenvector projection × exp(-λ·dt)
}
```

```bash
cargo run  # demo: heat diffusion on path, cycle, and complete graphs
```

## How It Works

1. Build the graph Laplacian L = D - A from adjacency matrix
2. Compute all eigenvalues and eigenvectors via Jacobi rotation
3. For each time step: project temperature onto eigenvectors, multiply by exp(-λᵢ·dt), reconstruct

The Jacobi method iteratively applies plane rotations to zero off-diagonal elements, converging to the eigenbasis. No external dependencies required.

## Testing

```bash
cargo test
cargo run
```

## Installation

```toml
[dependencies]
heat-spectral = { git = "https://github.com/SuperInstance/heat-spectral" }
```

## How It Fits

Part of the SuperInstance ecosystem:

- **heat-spectral** — Parabolic PDE on graphs (this repo)
- **[wave-conservation](https://github.com/SuperInstance/wave-conservation)** — Hyperbolic PDE (waves) on graphs
- **[graph-neural](https://github.com/SuperInstance/graph-neural)** — GNN spectral primitives

## License

MIT
