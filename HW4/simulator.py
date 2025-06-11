# Generate samples from a diffusion model using PyDDM

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Try to import PyDDM and its components
# If PyDDM is not installed, you can also install it manually
try:
    import pyddm as ddm
except ImportError as e:
    print(f"Failed to import PyDDM. Let's install it first...")
    os.system("pip install pyddm")
    import pyddm as ddm


def print_pyddm_version():
    print(f"PyDDM version: {ddm.__version__}")


def print_parameters(parameters):
    # PyDDM uses slightly different parameter conventions
    pyddm_x0 = parameters['boundary_sep_a'] * (parameters['relative_bias_b'] - 0.5)
    pyddm_bound = parameters['boundary_sep_a'] / 2.0
    # Print everything
    print(f"Parameters set for PyDDM simulation:")
    print(f"  + Drift Rate (v): {parameters['drift_rate_v']}")
    print(f"  + Boundary Separation (a): {parameters['boundary_sep_a']}")
    print(f"  + Non-decision Time (t0): {parameters['non_decision_t0']} s")
    print(f"  + Relative Bias (b): {parameters['relative_bias_b']}")
    print(f"PyDDM Conversions:")
    print(f"  -> PyDDM Bound Parameter (B for +/-B): {pyddm_bound}")
    print(f"  -> PyDDM Initial Condition (x0 relative to 0): {pyddm_x0}")


def print_summary(sim_rts, sim_choices):
    mean_upper_rt = np.mean(sim_rts[sim_choices == 1])
    mean_lower_rt = np.mean(sim_rts[sim_choices == 0])
    prop_upper = np.mean(sim_choices)

    print(f"Summary of PyDDM Simulation:")
    print(f"  + Mean Upper Boundary Response Time: {mean_upper_rt:.4f} seconds")
    print(f"  + Mean Lower Boundary Response Time: {mean_lower_rt:.4f} seconds")
    print(f"  + Proportion of Upper Boundary Choices: {prop_upper:.4f}")
    

def plot_rt_histogram(sim_rts, sim_choices, parameters, save_to_file=None):
    plt.figure(figsize=(10, 6))
    plt.hist(sim_rts[sim_choices == 1], 
             bins=50, 
             density=True, 
             alpha=0.75, 
             label="Simulated Upper Boundary RTs (PyDDM)", 
             color='mediumseagreen', 
             edgecolor='black')
    plt.hist(sim_rts[sim_choices == 0], 
             bins=50, 
             density=True, 
             alpha=0.75, 
             label="Simulated Lower Boundary RTs (PyDDM)", 
             color='lightcoral', 
             edgecolor='black')
    plt.axvline(np.mean(sim_rts[sim_choices == 1]), 
                color='red', 
                linestyle='dashed', 
                linewidth=3, 
                label=f"Mean Upper RT: {np.mean(sim_rts[sim_choices == 1]):.2f}s")
    plt.axvline(np.mean(sim_rts[sim_choices == 0]), 
                color='blue', 
                linestyle='dashed', 
                linewidth=3, 
                label=f"Mean Lower RT: {np.mean(sim_rts[sim_choices == 0]):.2f}s")
    title_str = (
        f"PyDDM: Simulated DDM RTs ({len(sim_rts)} Trials)\n"
        f"v={parameters['drift_rate_v']}, "
        f"a={parameters['boundary_sep_a']}, "
        f"t0={parameters['non_decision_t0']}, "
        f"b={parameters['relative_bias_b']}"
    )
    plt.title(title_str, fontsize=12)
    plt.xlabel("Response Time (s)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Better practice would be to prepare the figure in one function and then call that function
    # from two wrappers, one for saving and one for showing
    if save_to_file is not None:
        plt.savefig(save_to_file)
        print(f"\nResponse time histogram saved to '{save_to_file}'")
    else:
        plt.show()


def simulate_ddm(parameters, num_sim_trials):
    if parameters.get('simulation_seed', None) is not None:
        np.random.seed(parameters["simulation_seed"])

    # Make PyDDM parameters
    pyddm_drift = ddm.DriftConstant(drift=parameters["drift_rate_v"])
    pyddm_bound = ddm.BoundConstant(B=parameters['boundary_sep_a'] / 2.0)
    pyddm_x0 = ddm.ICPoint(x0=parameters['boundary_sep_a'] * (parameters['relative_bias_b'] - 0.5))
    pyddm_nondectime = ddm.OverlayNonDecision(nondectime=parameters["non_decision_t0"])
    pyddm_dt = 0.001

    # Create PyDDM Model instance
    model = ddm.Model(
        name    = "DDM_via_PyDDM",
        drift   = pyddm_drift,
        bound   = pyddm_bound,
        IC      = pyddm_x0,
        overlay = pyddm_nondectime,
        dt      = pyddm_dt
    )

    # Simulate multiple trials using model.solve() and solution.resample()
    solution = model.solve() 
    sample_obj = solution.resample(num_sim_trials)
    
    # Extract correct and incorrect RTs 
    correct_rts = sample_obj.choice_upper
    incorrect_rts = sample_obj.choice_lower
    
    # Combine into a single array
    rts = np.concatenate([correct_rts, incorrect_rts])
    choices = np.concatenate([np.ones(len(correct_rts)), np.zeros(len(incorrect_rts))])

    return rts, choices

#--- new function for creating delta plots ---
def create_delta_plots(parameters_base, parameters_contrast, num_trials=10000, save_to_file=None):
    """Create delta plots comparing individual parameter effects."""
    
    # Calculate percentiles for binning
    percentiles = [10, 30, 50, 70, 90]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Delta Plots: Effect of Parameter Changes on Choice and RT', fontsize=16)
    
    # Plot configurations - each comparison changes only ONE parameter
    param_pairs = [
        ('drift_rate_v', 'δ'),
        ('boundary_sep_a', 'α'), 
        ('non_decision_t0', 'τ'),
        ('relative_bias_b', 'β')
    ]
    
    # Colors for different percentiles
    colors = ['#8B0000', '#DAA520', '#808080', '#4169E1', '#9932CC']
    percentile_labels = ['10th percentile', '30th percentile', 'median', '70th percentile', '90th percentile']
    
    for idx, (param_name, param_symbol) in enumerate(param_pairs):
        ax = axes[idx // 2, idx % 2]
        
        # Create specific parameter sets for this comparison
        # Base condition uses base parameters
        current_base = parameters_base.copy()
        
        # Contrast condition: copy base parameters but change only the current parameter
        current_contrast = parameters_base.copy()
        current_contrast[param_name] = parameters_contrast[param_name]
        
        # Simulate both conditions
        rts_base, choices_base = simulate_ddm(current_base, num_trials)
        rts_contrast, choices_contrast = simulate_ddm(current_contrast, num_trials)
        
        # Calculate percentile values for both conditions
        base_percentiles = np.percentile(rts_base, percentiles)
        contrast_percentiles = np.percentile(rts_contrast, percentiles)
        
        # Calculate proportion of choices at each percentile for both conditions
        base_props = []
        contrast_props = []
        
        for i, perc in enumerate(percentiles):
            # Find trials close to this percentile RT
            if i == 0:
                base_mask = rts_base <= base_percentiles[i]
                contrast_mask = rts_contrast <= contrast_percentiles[i]
            else:
                base_mask = (rts_base > base_percentiles[i-1]) & (rts_base <= base_percentiles[i])
                contrast_mask = (rts_contrast > contrast_percentiles[i-1]) & (rts_contrast <= contrast_percentiles[i])
            
            if np.sum(base_mask) > 0:
                base_props.append(np.mean(choices_base[base_mask]))
            else:
                base_props.append(0.5)
                
            if np.sum(contrast_mask) > 0:
                contrast_props.append(np.mean(choices_contrast[contrast_mask]))
            else:
                contrast_props.append(0.5)
        
        base_props = np.array(base_props)
        contrast_props = np.array(contrast_props)
        
        # Calculate deltas (differences)
        deltas = contrast_props - base_props
        
        # Plot points and connecting lines
        for i, (perc, color, label) in enumerate(zip(percentiles, colors, percentile_labels)):
            # Plot base condition point
            ax.scatter(0, base_props[i], color=color, s=100, zorder=3)
            # Plot contrast condition point  
            ax.scatter(1, contrast_props[i], color=color, s=100, zorder=3, label=label)
            # Connect with dotted line
            ax.plot([0, 1], [base_props[i], contrast_props[i]], 
                   color=color, linestyle='--', alpha=0.7, linewidth=1.5)
            
            # Add delta annotation
            mid_y = (base_props[i] + contrast_props[i]) / 2
            ax.annotate(f'Δ{perc:.0f} = {deltas[i]:.2f}', 
                       xy=(0.5, mid_y), xytext=(1.2, mid_y),
                       fontsize=9, ha='left', va='center',
                       color=color, weight='bold')
        
        # Formatting
        ax.set_xlim(-0.2, 1.8)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Base', 'Contrast'])
        ax.set_ylabel('Proportion Correct Choice', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.3)
        
        # Add parameter values to title
        title = f"Effect of {param_symbol}\n"
        title += f"Base {param_symbol}={current_base[param_name]:.2f}, "
        title += f"Contrast {param_symbol}={current_contrast[param_name]:.2f}"
        ax.set_title(title, fontsize=11)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if save_to_file is not None:
        plt.savefig(save_to_file, dpi=300, bbox_inches='tight')
        print(f"\nDelta plots saved to '{save_to_file}'")
    else:
        plt.show()

# --- Main execution block for demonstration ---
if __name__ == "__main__":
    print("--- PyDDM Simulator Demonstration ---")

    print_pyddm_version()

    parameters = {
        "drift_rate_v": 1.0,
        "boundary_sep_a": 2.0,
        "non_decision_t0": 0.25,
        "relative_bias_b": 0.35
    }

    # Base parameters
    parameters_base = {
        "drift_rate_v": 2.0,    # δ
        "boundary_sep_a": 1.0,   # α
        "non_decision_t0": 0.25, # τ
        "relative_bias_b": 0.35  # β
    }

    # Contrast parameters
    parameters_contrast = {
        "drift_rate_v": 1.0,    # δ
        "boundary_sep_a": 2.0,   # α
        "non_decision_t0": 0.50, # τ
        "relative_bias_b": 0.65  # β
    }

    # Create and save delta plots
    save_plot_name = Path(__file__).parent.parent / "figures" / "pyddm_delta_plots.png"
    create_delta_plots(parameters_base, parameters_contrast, num_trials=10000, save_to_file=save_plot_name)

    print_parameters(parameters)

    sim_rts, sim_choices = simulate_ddm(parameters, 10000)

    print_summary(sim_rts, sim_choices)

    save_plot_name = Path(__file__).parent.parent / "figures" / "pyddm_simulation_rts_histogram.png"
    plot_rt_histogram(sim_rts, 
                      sim_choices,
                      parameters, 
                      save_to_file=save_plot_name)

