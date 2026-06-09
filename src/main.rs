use heat_spectral::*;

fn main() {
    println!("=== Heat Spectral: Heat Diffusion on Graphs ===\n");

    println!("--- Experiment 1: Equilibration Time ---");
    let adj = cycle_graph(8);
    let report = verify_equilibration_time(&adj);
    println!("Graph: Cycle(8)");
    println!("  CR: {:.4}", report.cr);
    println!(
        "  Predicted eq time (1/λ₂): {:.4}",
        report.predicted_eq_time
    );
    println!("  Measured eq time: {:.4}", report.equilibration_time);
    println!(
        "  Heat conservation error: {:.2e}",
        report.heat_conservation_error
    );
    println!();

    println!("--- Experiment 2: CR vs Equilibration Speed ---");
    let results = cr_vs_equilibration();
    println!("{:<15} {:>10} {:>20}", "Graph", "CR", "1/λ₂ (eq time)");
    let names = ["path-6", "cycle-6", "complete-6", "star-6", "barbell-4"];
    for (i, (cr, eq)) in results.iter().enumerate() {
        println!("{:<15} {:>10.4} {:>20.4}", names[i], cr, eq);
    }
    println!();

    println!("--- Experiment 3: Spectral Filtering ---");
    let n = 8;
    let adj = cycle_graph(n);
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin()
                + 0.3 * (6.0 * std::f64::consts::PI * i as f64 / n as f64).sin()
        })
        .collect();
    let short = spectral_filtering(&signal, &adj, 0.05);
    let long = spectral_filtering(&signal, &adj, 2.0);
    println!(
        "Signal:       {:?}",
        signal
            .iter()
            .map(|x| format!("{:.3}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "Short (0.05): {:?}",
        short
            .iter()
            .map(|x| format!("{:.3}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "Long  (2.00): {:?}",
        long.iter()
            .map(|x| format!("{:.3}", x))
            .collect::<Vec<_>>()
    );
    println!();

    println!("--- Experiment 4: Heat Anomaly Detection ---");
    let adj = cycle_graph(8);
    let normal_dist = heat_anomaly_detection(&adj, 0);
    println!("Cycle(8): node 0 deviation: {:.6}", normal_dist);
    let adj2 = barbell_graph(4);
    let d1 = heat_anomaly_detection(&adj2, 0);
    let d2 = heat_anomaly_detection(&adj2, 3);
    println!("Barbell: node 0 deviation: {:.6}", d1);
    println!("Barbell: bridge node 3 deviation: {:.6}", d2);
    println!();

    println!("--- Experiment 5: Thermal Image (Path-4) ---");
    let adj = path_graph(4);
    let thermal = thermal_image(&adj);
    for (i, row) in thermal.iter().enumerate() {
        print!("Source {}: ", i);
        for &t in row {
            print!("{:.4} ", t);
        }
        println!();
    }
    println!();

    println!("--- Diffusion vs Wave Comparison ---");
    for (adj, name) in [
        (path_graph(6), "Path(6)"),
        (cycle_graph(6), "Cycle(6)"),
        (complete_graph(6), "Complete(6)"),
    ] {
        let cmp = compare_diffusion_wave(&adj, name);
        println!(
            "{:<12} CR={:.4} heat_eq={:.4} wave_half={} ratio={:.4}",
            cmp.graph_name,
            cmp.cr,
            cmp.heat_equilibration_time,
            cmp.wave_coherence_halflife,
            cmp.ratio
        );
    }
}
