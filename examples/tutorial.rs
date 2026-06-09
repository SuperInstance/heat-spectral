//! Tutorial: Heat Diffusion on Graphs — A Guided Walkthrough
//!
//! Run with: `cargo run --example tutorial`

use heat_spectral::*;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   Heat Spectral Tutorial: Understanding Graph Diffusion     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ── Lesson 1: The Graph Laplacian ────────────────────────────
    println!("━━━ Lesson 1: The Graph Laplacian ━━━\n");
    println!("The Laplacian L = D - A encodes graph topology in matrix form.");
    println!("For a path graph 0-1-2-3, the Laplacian is:\n");
    println!("  L = | 1  -1   0   0 |");
    println!("      |-1   2  -1   0 |");
    println!("      | 0  -1   2  -1 |");
    println!("      | 0   0  -1   1 |\n");
    println!("Each row sums to zero (conservation). Diagonal = degree.\n");

    let adj = path_graph(4);
    let state = HeatState::new(&adj);

    println!("Eigenvalues: ");
    for (i, &lambda) in state.eigenvalues.iter().enumerate() {
        println!("  λ_{} = {:.6}", i + 1, lambda);
    }
    println!("\n  λ₁ ≈ 0: the constant eigenvector (uniform heat → no change)");
    println!("  λ₂ = {:.4}: the Fiedler value — controls equilibration speed\n", state.eigenvalues[1]);

    // ── Lesson 2: Point Source Diffusion ─────────────────────────
    println!("━━━ Lesson 2: Point Source Diffusion ━━━\n");
    println!("Place a unit heat pulse at node 0 and watch it spread:\n");

    let adj = cycle_graph(8);
    let mut sim = HeatState::new(&adj);
    sim.set_heat(0, 1.0);

    println!("  t=0.00: {:?}", format_temps(&sim));
    for step in 1..=5 {
        sim.step(0.2);
        println!("  t={:.2}: {:?}", 0.2 * step as f64, format_temps(&sim));
    }
    println!("\n  Heat spreads outward, then approaches uniform 1/8 = {:.4}",
             1.0 / 8.0);

    // ── Lesson 3: Conservation Ratio ─────────────────────────────
    println!("\n━━━ Lesson 3: The Conservation Ratio CR = λ₂/λₙ ━━━\n");

    let graphs: Vec<(&str, Vec<Vec<f64>>)> = vec![
        ("Path(8)",     path_graph(8)),
        ("Cycle(8)",    cycle_graph(8)),
        ("Star(8)",     star_graph(8)),
        ("Complete(8)", complete_graph(8)),
        ("Barbell(4)",  barbell_graph(4)),
    ];

    println!("  {:<15} {:>8} {:>15} {:>12}", "Graph", "CR", "1/λ₂ (eq time)", "λₙ");
    println!("  {}", "─".repeat(55));
    for (name, adj) in &graphs {
        let state = HeatState::new(adj);
        let cr = state.cr();
        let eq = state.equilibration_time();
        let ln = state.eigenvalues.last().unwrap();
        println!("  {:<15} {:>8.4} {:>15.4} {:>12.4}", name, cr, eq, ln);
    }
    println!("\n  High CR → fast mixing. Low CR → bottlenecks slow diffusion.");

    // ── Lesson 4: Spectral Filtering ─────────────────────────────
    println!("\n━━━ Lesson 4: Spectral Filtering (Graph Gaussian Blur) ━━━\n");

    let n = 10;
    let adj = cycle_graph(n);

    // Create a noisy signal
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let clean = (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin();
            let noise = if i % 3 == 0 { 0.5 } else { 0.0 };
            clean + noise
        })
        .collect();

    println!("  Original signal: {:?}",
        signal.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>());

    let smoothed = spectral_filtering(&signal, &adj, 0.1);
    println!("  After diffusion: {:?}",
        smoothed.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>());

    let very_smooth = spectral_filtering(&signal, &adj, 2.0);
    println!("  Heavy diffusion: {:?}",
        very_smooth.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>());
    println!("\n  Short diffusion = high-pass. Long diffusion = heavy smoothing.");

    // ── Lesson 5: Exact vs Euler ─────────────────────────────────
    println!("\n━━━ Lesson 5: Exact Steps vs Forward Euler ━━━\n");

    let adj = complete_graph(5);
    let mut exact = HeatState::new(&adj);
    let mut euler = HeatState::new(&adj);
    exact.set_heat(0, 1.0);
    euler.set_heat(0, 1.0);

    println!("  Exact spectral step with dt=10.0 (huge!):");
    for _ in 0..10 {
        exact.step(10.0);
    }
    println!("    Total heat: {:.10} (should be 1.0)", exact.total_heat());
    println!("    Max temp:   {:.10}", exact.temperature().iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    println!("\n  Forward Euler with dt=1.0 (large):");
    let mut blew_up = false;
    for i in 0..50 {
        euler.step_euler(1.0);
        if euler.temperature().iter().any(|t| t.is_nan() || t.abs() > 1e10) {
            println!("    BLEW UP at step {}!", i + 1);
            blew_up = true;
            break;
        }
    }
    if !blew_up {
        println!("    Survived (total heat: {:.6})", euler.total_heat());
    }
    println!("\n  Moral: exact spectral steps are ALWAYS stable. Euler is not.");

    println!("\n✅ Tutorial complete! See examples/advanced.rs for real-world use cases.");
}

fn format_temps(sim: &HeatState) -> Vec<String> {
    sim.temperature()
        .iter()
        .map(|x| format!("{:.4}", x))
        .collect()
}
