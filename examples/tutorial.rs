//! # Heat Spectral Tutorial
//!
//! A progressive, hands-on tour of spectral graph theory through heat diffusion.
//! Each lesson teaches ONE concept with clear println output.
//!
//! Run with: cargo run --example tutorial

use heat_spectral::*;
use std::f64::consts::PI;

fn separator(title: &str) {
    println!("\n{}", "═".repeat(60));
    println!("  Lesson: {}", title);
    println!("{}\n", "═".repeat(60));
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Heat Spectral — Progressive Tutorial                   ║");
    println!("║  Laplacian eigenvalues, heat diffusion, spectral gaps    ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    lesson_01_graph_laplacian();
    lesson_02_eigenvalues_and_connectivity();
    lesson_03_heat_diffusion_exact();
    lesson_04_conservation_ratio();
    lesson_05_spectral_filtering();
    lesson_06_graph_families_comparison();
    lesson_07_anomaly_detection();
    lesson_08_heat_vs_waves();
}

// ─── Lesson 1: Graph Laplacian ───────────────────────────────

fn lesson_01_graph_laplacian() {
    separator("1 — The Graph Laplacian: L = D − A");

    println!("The Laplacian captures the topology of a graph.");
    println!("L = D - A where D = degree matrix, A = adjacency matrix.");
    println!();

    let adj = path_graph(4);
    let state = HeatState::new(&adj);

    println!("Path graph on 4 nodes: 0─1─2─3");
    println!();
    println!("Laplacian L:");
    for (i, row) in state.laplacian.iter().enumerate() {
        print!("  [");
        for (j, &val) in row.iter().enumerate() {
            if j > 0 { print!(", "); }
            print!("{:5.1}", val);
        }
        println!("]  (row {})", i);
    }
    println!();

    println!("Key properties:");
    let mut row_sums_ok = true;
    for (_i, row) in state.laplacian.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        if sum.abs() > 1e-10 { row_sums_ok = false; }
    }
    println!("  • Every row sums to 0: {} ✓", row_sums_ok);
    println!("  • Diagonal entries = degree of each node");
    println!("  • Off-diagonal entries = -1 if connected, 0 otherwise");
    println!("  • L is symmetric, positive semi-definite");
}

// ─── Lesson 2: Eigenvalues & Connectivity ────────────────────

fn lesson_02_eigenvalues_and_connectivity() {
    separator("2 — Laplacian Eigenvalues & Algebraic Connectivity");

    let graphs: Vec<(&str, Vec<Vec<f64>>)> = vec![
        ("Path(5)", path_graph(5)),
        ("Cycle(5)", cycle_graph(5)),
        ("Complete(5)", complete_graph(5)),
        ("Star(5)", star_graph(5)),
    ];

    println!("Laplacian eigenvalues (sorted ascending):");
    println!("  {:<15} {:>10} {:>10} {:>10}", "Graph", "λ₁", "λ₂", "λₙ");
    println!("  {}+{}+{}+{}", "-".repeat(15), "-".repeat(10), "-".repeat(10), "-".repeat(10));

    for (name, adj) in &graphs {
        let state = HeatState::new(adj);
        let n = state.eigenvalues.len();
        println!("  {:<15} {:>10.4} {:>10.4} {:>10.4}",
                 name,
                 state.eigenvalues[0],
                 state.eigenvalues[1],
                 state.eigenvalues[n - 1]);
    }
    println!();

    println!("Key spectral graph theory theorems:");
    println!("  • λ₁ = 0 always (constant vector is eigenvector)");
    println!("  • λ₂ > 0 ⟺ graph is connected (Fiedler value)");
    println!("  • λ₂ is the 'algebraic connectivity' — how well-connected");
    println!("  • Larger λ₂ = faster mixing = better connected");
    println!();

    let cycle = HeatState::new(&cycle_graph(8));
    println!("Cycle(8) eigenvalue spectrum:");
    for (i, &lambda) in cycle.eigenvalues.iter().enumerate() {
        println!("  λ_{} = {:8.4}{}", i + 1, lambda, if i == 0 { " ← zero mode" } else { "" });
    }
}

// ─── Lesson 3: Exact Spectral Heat Diffusion ────────────────

fn lesson_03_heat_diffusion_exact() {
    separator("3 — Exact Spectral Heat Diffusion");

    let adj = cycle_graph(8);
    let mut sim = HeatState::new(&adj);
    sim.set_heat(0, 1.0);

    println!("Heat pulse at node 0 on Cycle(8).");
    println!("Exact solution: u(t) = Σ cᵢ · exp(-λᵢt) · vᵢ");
    println!();

    println!("  Step | Heat@0   Heat@1   Heat@2   Heat@3   Heat@4  | Total  Variance");
    println!("  -----+------------------------------------------------+------------------");
    println!("    {:2} | {:.4}   {:.4}   {:.4}   {:.4}   {:.4}  | {:.4}  {:.6}",
             0, sim.temperature()[0], sim.temperature()[1], sim.temperature()[2],
             sim.temperature()[3], sim.temperature()[4], sim.total_heat(), sim.variance());

    let dt = 0.1;
    for step in [5, 10, 20, 50, 100] {
        let current_step = step;
        // Reset and step forward
        sim.set_heat(0, 1.0);
        for _ in 0..current_step {
            sim.step(dt);
        }
        print!("  {:3} | {:.4}   {:.4}   {:.4}   {:.4}   {:.4}  | {:.4}  {:.6}",
               current_step, sim.temperature()[0], sim.temperature()[1], sim.temperature()[2],
               sim.temperature()[3], sim.temperature()[4], sim.total_heat(), sim.variance());
        println!();
    }
    println!();

    println!("Observations:");
    println!("  • Total heat is CONSERVED (exact solution, no drift)");
    println!("  • Variance decreases monotonically → heat spreads evenly");
    println!("  • High-frequency modes die fast, low-frequency modes persist");
}

// ─── Lesson 4: Conservation Ratio ────────────────────────────

fn lesson_04_conservation_ratio() {
    separator("4 — Conservation Ratio: CR = λ₂/λₙ");

    println!("CR captures mixing quality in a single number:");
    println!("  CR > 0.7 → fast uniform mixing (expanders, complete graphs)");
    println!("  CR ~ 0.3 → moderate mixing (cycles, regular graphs)");
    println!("  CR < 0.1 → poor mixing, bottlenecks (paths, barbell graphs)");
    println!();

    let graphs: Vec<(&str, Vec<Vec<f64>>)> = vec![
        ("Complete(6)", complete_graph(6)),
        ("Cycle(6)", cycle_graph(6)),
        ("Star(6)", star_graph(6)),
        ("Path(6)", path_graph(6)),
        ("Barbell(4)", barbell_graph(4)),
    ];

    println!("  {:<15} {:>8} {:>12} {:>12}", "Graph", "CR", "1/λ₂ (eq)", "λₙ");
    println!("  {}+{}+{}+{}", "-".repeat(15), "-".repeat(8), "-".repeat(12), "-".repeat(12));

    for (name, adj) in &graphs {
        let state = HeatState::new(adj);
        let cr = state.cr();
        let eq_time = state.equilibration_time();
        let n = state.eigenvalues.len();
        let lambda_n = state.eigenvalues[n - 1];
        println!("  {:<15} {:>8.4} {:>12.4} {:>12.4}", name, cr, eq_time, lambda_n);
    }
    println!();

    let complete = HeatState::new(&complete_graph(6));
    let path = HeatState::new(&path_graph(6));
    println!("  Complete graph CR ({:.4}) >> Path CR ({:.4})",
             complete.cr(), path.cr());
    println!("  → Complete graph equilibrates {:.1}× faster",
             path.equilibration_time() / complete.equilibration_time());
}

// ─── Lesson 5: Spectral Filtering ────────────────────────────

fn lesson_05_spectral_filtering() {
    separator("5 — Spectral Filtering: Graph Signal Processing");

    let n = 10;
    let adj = cycle_graph(n);

    // Create a clean signal + noise
    let clean: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / n as f64).sin())
        .collect();
    let noise: Vec<f64> = vec![0.3, -0.5, 0.2, 0.4, -0.3, 0.6, -0.2, 0.5, -0.4, 0.3];
    let noisy: Vec<f64> = clean.iter().zip(noise.iter()).map(|(c, n)| c + n).collect();

    println!("Graph signal = clean sinusoid + noise on Cycle(10):");
    println!();

    let fmt = |v: &[f64]| -> String {
        v.iter().map(|x| format!("{:+.3}", x)).collect::<Vec<_>>().join(", ")
    };

    println!("  Clean:  [{}]", fmt(&clean));
    println!("  Noisy:  [{}]", fmt(&noisy));
    println!();

    // Short diffusion = gentle smoothing (preserves low-freq, removes high-freq)
    let short = spectral_filtering(&noisy, &adj, 0.05);
    let medium = spectral_filtering(&noisy, &adj, 0.3);
    let long = spectral_filtering(&noisy, &adj, 2.0);

    println!("  After t=0.05 diffusion (gentle):  [{}]", fmt(&short));
    println!("  After t=0.30 diffusion (medium):  [{}]", fmt(&medium));
    println!("  After t=2.00 diffusion (heavy):   [{}]", fmt(&long));
    println!();

    // Compute reconstruction error
    let error = |a: &[f64], b: &[f64]| -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
    };
    println!("  Reconstruction error vs clean signal:");
    println!("    Noisy:   {:.4}", error(&noisy, &clean));
    println!("    t=0.05:  {:.4}", error(&short, &clean));
    println!("    t=0.30:  {:.4}", error(&medium, &clean));
    println!("    t=2.00:  {:.4}", error(&long, &clean));
    println!();

    println!("Heat diffusion is the graph analog of Gaussian blur:");
    println!("  Short time → preserves structure, removes high-freq noise");
    println!("  Long time → oversmooths toward uniform distribution");
}

// ─── Lesson 6: Graph Families Comparison ─────────────────────

fn lesson_06_graph_families_comparison() {
    separator("6 — Full Comparison: CR vs Equilibration Across Graphs");

    println!("Measuring how CR predicts equilibration speed...");
    println!();

    let results = cr_vs_equilibration();
    let names = ["Path(6)", "Cycle(6)", "Complete(6)", "Star(6)", "Barbell(4)"];

    println!("  {:<15} {:>8} {:>18} {:>18} {:>15}",
             "Graph", "CR", "Predicted 1/λ₂", "Measured eq time", "Conserv. err");
    println!("  {}+{}+{}+{}+{}",
             "-".repeat(15), "-".repeat(8), "-".repeat(18), "-".repeat(18), "-".repeat(15));

    let graphs_for_report: Vec<Vec<Vec<f64>>> = vec![
        path_graph(6), cycle_graph(6), complete_graph(6), star_graph(6), barbell_graph(4),
    ];
    for (i, (cr, eq_time)) in results.iter().enumerate() {
        let report = verify_equilibration_time(&graphs_for_report[i]);
        println!("  {:<15} {:>8.4} {:>18.4} {:>18.4} {:>15.2e}",
                 names[i], cr, eq_time, report.equilibration_time,
                 report.heat_conservation_error);
    }
    println!();

    println!("Pattern: Higher CR → Faster equilibration");
    println!("The prediction 1/λ₂ is a good estimate of actual mixing time.");
}

// ─── Lesson 7: Anomaly Detection ─────────────────────────────

fn lesson_07_anomaly_detection() {
    separator("7 — Thermal Anomaly Detection");

    println!("Each node's 'thermal fingerprint' reveals its role in the graph.");
    println!("Anomalous nodes (bridges, hubs) produce unusual patterns.");
    println!();

    // Cycle graph — all nodes are equivalent
    println!("── Cycle(8): All nodes equivalent ──");
    let cycle_adj = cycle_graph(8);
    for node in [0, 3, 7] {
        let dev = heat_anomaly_detection(&cycle_adj, node);
        println!("  Node {} deviation: {:.6}", node, dev);
    }
    println!("  → All nodes have similar deviation (symmetric graph)");
    println!();

    // Barbell graph — bridge node is anomalous
    println!("── Barbell(4): Bridge node is anomalous ──");
    let barbell_adj = barbell_graph(4);
    println!("  Two K₄ cliques connected by a single bridge edge");
    println!("  Nodes 0-3: left clique, Nodes 4-7: right clique");
    println!("  Node 3↔4 is the bridge");
    println!();

    for node in [0, 3, 4, 7] {
        let dev = heat_anomaly_detection(&barbell_adj, node);
        println!("  Node {} deviation: {:.6}{}", node, dev,
                 if node == 3 || node == 4 { " ← bridge node" } else { "" });
    }
    println!();

    // Thermal image
    println!("── Thermal Image for Path(4) ──");
    let path_adj = path_graph(4);
    let thermal = thermal_image(&path_adj);
    println!("  Each row = steady-state from a single source node:");
    for (i, row) in thermal.iter().enumerate() {
        print!("  Source {}: ", i);
        for &t in row {
            print!("{:.4} ", t);
        }
        println!();
    }
}

// ─── Lesson 8: Heat Diffusion vs Wave Propagation ────────────

fn lesson_08_heat_vs_waves() {
    separator("8 — Heat Diffusion vs Wave Propagation");

    println!("Heat equation: ∂u/∂t = -Lu  (dissipative, converges)");
    println!("Wave equation: ∂²u/∂t² = -Lu (conservative, oscillates)");
    println!();

    let graphs: Vec<(&str, Vec<Vec<f64>>)> = vec![
        ("Path(6)", path_graph(6)),
        ("Cycle(6)", cycle_graph(6)),
        ("Complete(6)", complete_graph(6)),
        ("Star(6)", star_graph(6)),
        ("Barbell(4)", barbell_graph(4)),
    ];

    println!("  {:<12} {:>8} {:>14} {:>14} {:>8}",
             "Graph", "CR", "Heat eq time", "Wave halflife", "Ratio");
    println!("  {}+{}+{}+{}+{}",
             "-".repeat(12), "-".repeat(8), "-".repeat(14), "-".repeat(14), "-".repeat(8));

    for (name, adj) in &graphs {
        let report = compare_diffusion_wave(adj, name);
        println!("  {:<12} {:>8.4} {:>14.4} {:>14} {:>8.4}",
                 report.graph_name, report.cr,
                 report.heat_equilibration_time,
                 report.wave_coherence_halflife,
                 report.ratio);
    }
    println!();

    println!("Key insight:");
    println!("  • Heat always equilibrates (monotone decay of variance)");
    println!("  • Waves oscillate forever (energy is conserved)");
    println!("  • The ratio reveals how graph topology affects each differently");
    println!("  • Barbell shows the largest gap — bottlenecks trap waves longer");
}
