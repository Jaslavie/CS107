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
def create_delta_plots(parameters_base, parameters_contrast, num_trials = 10000, save_to_file=None):
    # delta plots comparing two DDM (base and contrast) parameter sets
    # rts = response time
    # choices = choice (1 = correct, 0 = incorrect)
    # the base condition is the original condition and contrast is the condition where the delta is applied

    #* ----- Setup -----
    # simulate both conditions
    rts_base, choices_base = simulate_ddm(parameters_base, num_trials) # base condition
    rts_contrast, choices_contrast = simulate_ddm(parameters_contrast, num_trials) # contrast condition

    # calculate percentiles for binning
    # each bin calculates the mean choice and RT for 10% of the data
    percentiles = [10, 30, 50, 70, 90]

    # calculate the mean choice and RT for each percentile bin
    # returns two arrays: bin_means_rt and bin_means_choice
    def calculate_bin_stats(rts, choices, rt_bins):
        bin_means_rt = []
        bin_means_choice = []

        for i in range(len(rt_bins) - 1):
            # mask is a boolean array used to select data within the current bin
            # the selected data is used to calculate the mean choice and RT for the bin
            mask = (rts >= rt_bins[i]) & (rts < rt_bins[i+1])

            if np.sum(mask) > 0:
                bin_means_rt.append(np.mean(rts[mask]))
                bin_means_choice.append(np.mean(choices[mask]))
            else:
                # If no data in bin, use the midpoint of bin for RT and 0.5 for choice
                bin_means_rt.append((rt_bins[i] + rt_bins[i+1]) / 2)
                bin_means_choice.append(0.5)
        
        return np.array(bin_means_rt), np.array(bin_means_choice)

    #* ----- Plot -------- generated by claude
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Delta Plots: Effect of Parameter Changes on Choice and RT', fontsize=16)
    
    # Plot configurations
    param_pairs = [
        ('drift_rate_v', 'δ'),
        ('boundary_sep_a', 'α'),
        ('non_decision_t0', 'τ'),
        ('relative_bias_b', 'β')
    ]
    
    for idx, (param_name, param_symbol) in enumerate(param_pairs):
        ax = axes[idx // 2, idx % 2]
        
        # Get RT percentile bins for both conditions combined
        all_rts = np.concatenate([rts_base, rts_contrast])
        rt_bins = np.percentile(all_rts, percentiles)
        rt_bins = np.insert(rt_bins, 0, 0)  # Add 0 as first bin
        rt_bins = np.append(rt_bins, np.inf)  # Add inf as last bin
        
        # Calculate bin statistics
        base_rt, base_choice = calculate_bin_stats(rts_base, choices_base, rt_bins)
        contrast_rt, contrast_choice = calculate_bin_stats(rts_contrast, choices_contrast, rt_bins)
        
        # Calculate effect size (difference between conditions)
        effect_size = contrast_choice - base_choice
        
        # Plot
        ax.plot(base_rt, effect_size, 'o-', color='purple', linewidth=2, markersize=8)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlabel('Mean RT (s)', fontsize=10)
        ax.set_ylabel('Effect Size (Contrast - Base)', fontsize=10)
        
        # Add parameter values to title
        title = f"Effect of {param_symbol}\n"
        title += f"Base {param_symbol}={parameters_base[param_name]:.2f}, "
        title += f"Contrast {param_symbol}={parameters_contrast[param_name]:.2f}"
        ax.set_title(title, fontsize=12)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
    plt.tight_layout()
    
    if save_to_file is not None:
        plt.savefig(save_to_file)
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

