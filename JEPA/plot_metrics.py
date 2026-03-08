#!/usr/bin/env python3
"""
System 2 EBM JEPA Telemetry Visualizer.
Reads the live CSV log file, renders a stacked metrics graph,
and prints a summary table of the best metrics.

Usage:
    python JEPA/plot_metrics.py
"""
import os
import csv
import matplotlib.pyplot as plt

def plot_metrics(csv_path="jepa_logs/training_metrics.csv"):
    if not os.path.exists(csv_path):
        print(f"❌ Could not find {csv_path}. Is the training script running?")
        return

    steps, energy, sim, var, cov = [], [], [], [], []

    print(f"📊 Reading live telemetry from {csv_path}...")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(row["Global_Step"]))
                energy.append(float(row["Total_Energy"]))
                sim.append(float(row["Sim"]))
                var.append(float(row["Var"]))
                cov.append(float(row["Cov"]))
            except ValueError:
                # Silently skip any corrupted or partially written rows
                continue 

    if not steps:
        print("⚠️ CSV file exists but contains no valid data rows yet.")
        return

    print(f"📈 Plotting {len(steps)} data points...")
    
    # --- Print Best Stats Table ---
    print("\n" + "="*40)
    print("🏆 CURRENT BEST TRAINING METRICS 🏆")
    print("="*40)
    print(f"| {'Metric':<15} | {'Best Value':<18} |")
    print("-" * 40)
    print(f"| {'Total Energy':<15} | {min(energy):<18.4f} |")
    print(f"| {'Sim (MSE)':<15} | {min(sim):<18.4f} |")
    print(f"| {'Var (Collapse)':<15} | {min(var):<18.4f} |")
    print(f"| {'Cov (Tangle)':<15} | {min(cov):<18.4f} |")
    print("="*40 + "\n")
    
    # Create a 4-panel stacked plot
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # Panel 1: Total Energy
    axs[0].plot(steps, energy, color='black', linewidth=1.5)
    axs[0].set_title("Total EBM Energy (Lower is Better)")
    axs[0].set_ylabel("Energy")
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Panel 2: Sim (Invariance)
    axs[1].plot(steps, sim, color='blue', linewidth=1.5)
    axs[1].set_title("Prediction Error [Sim] (Lower is Better)")
    axs[1].set_ylabel("MSE")
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # Panel 3: Var (Variance)
    axs[2].plot(steps, var, color='red', linewidth=1.5)
    axs[2].set_title("Representation Collapse Penalty [Var]")
    axs[2].set_ylabel("Penalty")
    axs[2].grid(True, linestyle='--', alpha=0.6)

    # Panel 4: Cov (Covariance)
    axs[3].plot(steps, cov, color='green', linewidth=1.5)
    axs[3].set_title("Covariance (Decorrelation) Penalty [Cov]")
    axs[3].set_xlabel("Global Training Step")
    axs[3].set_ylabel("Penalty")
    axs[3].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    # Save the output
    out_path = "jepa_logs/live_training_curve.png"
    plt.savefig(out_path, dpi=300)
    print(f"✅ Success! Saved high-resolution plot to {out_path}")

if __name__ == "__main__":
    plot_metrics()