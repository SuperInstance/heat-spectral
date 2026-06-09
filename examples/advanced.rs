//! Advanced Example: Real-World Applications of Heat Spectral Analysis
//!
//! Demonstrates network analysis, anomaly detection, thermal imaging,
//! and the diffusion-vs-wave comparison on realistic graph topologies.
//!
//! Run with: `cargo run --example advanced`

use heat_spectral::*;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   Advanced Heat Spectral: Real-World Applications       ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // ── Application 1: Network Bottleneck Detection ──────────────
    println!("━━━ Application 1: Finding Network Bottlenecks ━━━\n");
    println!("Imagine a network with two well-connected clusters linked");
    println!("by a single bridge. Which node is the bottleneck?\n");

    let adj = barbell_graph(5); // Two 5-cliques connected by bridge
    let n = adj.len();

    println!("  Barbell graph: two 5-cliques + bridge between nodes 4 ↔ 5");
    println!("  Computing anomaly scores for each node...\n");

    println!("  {:>6} {:>12} {:>8}", "Node", "Deviation", "Role");
    println!("  {}", "─".repeat(30));
    for node in 0..n {
        let dev = heat_anomaly_detection(&adj, node);
        let role = if node == 4 || node == 5 {
            "BRIDGE"
        } else if node < 5 {
            "clique A"
        } else {
            "clique B"
        };
        println!("  {:>6} {:>12.6} {:>8}", node, dev, role);
    }
    println!("\n  Bridge nodes have distinct thermal signatures — heat lingers\n  at the bottleneck before crossing to the other clique.\n");

    // ── Application 2: Thermal Image Visualization ───────────────
    println!("━━━ Application 2: Thermal Image of Graph Communication ━━━\n");
    println!("Each row shows how heat from one source distributes across the graph.");
    println!("After sufficient diffusion, each row approximates the stationary\n");
    println!("distribution — how \"reachable\" each node is from the source.\n");

    let adj = star_graph(5);
    let thermal = thermal_image(&adj);

    println!("  Star(5) thermal image (source → destination):\n");
    print!("  {:>10}", "");
    for j in 0..thermal.len() {
        print!(" {:>7}", format!("n{}", j));
    }
    println!();
    for (i, row) in thermal.iter().enumerate() {
        print!("  source={}: ", i);
        for &t in row {
            print!(" {:>7.4}", t);
        }
        println!();
    }
    println!("\n  Node 0 (hub) distributes heat uniformly. Leaf nodes concentrate\n  heat at the hub first, then spread outward.\n");

    // ── Application 3: Spectral Signal Denoising ─────────────────
    println!("━━━ Application 3: Signal Denoising on Graphs ━━━\n");

    let n = 12;
    let adj = cycle_graph(n);

    // True signal: smooth sinusoid
    let truth: Vec<f64> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin())
        .collect();

    // Add impulse noise
    let mut noisy = truth.clone();
    noisy[2] += 2.0;
    noisy[7] -= 1.5;
    noisy[9] += 1.8;

    let noise_l2 = l2_dist(&noisy, &truth);
    println!("  Signal + impulse noise. L₂ error from truth: {:.4}", noise_l2);

    // Find optimal diffusion time by scanning
    println!("\n  Scanning diffusion times to find optimal denoising:\n");
    println!("  {:>10} {:>12} {:>10}", "Diff. time", "L₂ error", "vs noisy");
    println!("  {}", "─".repeat(35));

    let mut best_time = 0.0;
    let mut best_err = f64::INFINITY;
    for &dt in &[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0] {
        let filtered = spectral_filtering(&noisy, &adj, dt);
        let err = l2_dist(&filtered, &truth);
        let vs = if err < noise_l2 { "✓ better" } else { "✗ worse" };
        println!("  {:>10.3} {:>12.4} {:>10}", dt, err, vs);
        if err < best_err {
            best_err = err;
            best_time = dt;
        }
    }
    println!("\n  Optimal diffusion time: {:.3} (error: {:.4}, improvement: {:.1}%)",
             best_time, best_err,
             (1.0 - best_err / noise_l2) * 100.0);

    // ── Application 4: Diffusion vs Wave Physics ─────────────────
    println!("\n━━━ Application 4: Heat Diffusion vs Wave Propagation ━━━\n");
    println!("Heat dissipates (irreversible). Waves oscillate (reversible).");
    println!("How do different topologies affect these two processes?\n");

    println!("  {:<15} {:>6} {:>12} {:>10} {:>10}",
             "Graph", "CR", "Heat eq (s)", "Wave half", "Ratio");
    println!("  {}", "─".repeat(55));

    let topologies: Vec<(&str, Vec<Vec<f64>>)> = vec![
        ("Path(8)",     path_graph(8)),
        ("Cycle(8)",    cycle_graph(8)),
        ("Complete(8)", complete_graph(8)),
        ("Star(8)",     star_graph(8)),
        ("Barbell(4)",  barbell_graph(4)),
    ];

    for (name, adj) in &topologies {
        let report = compare_diffusion_wave(adj, name);
        println!("  {:<15} {:>6.4} {:>12.4} {:>10} {:>10.4}",
                 report.graph_name,
                 report.cr,
                 report.heat_equilibration_time,
                 report.wave_coherence_halflife,
                 report.ratio);
    }
    println!("\n  Ratio = wave coherence halflife / heat equilibration time.");
    println!("  Higher ratio → waves persist much longer than heat takes to equilibrate.");
    println!("  Complete graphs: waves and heat both behave similarly (fast mixing).");
    println!("  Path graphs: waves bounce back and forth, heat slowly diffuses.");

    println!("\n✅ Advanced examples complete.");
}

fn l2_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}
