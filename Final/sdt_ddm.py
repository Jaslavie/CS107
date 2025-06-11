"""
Signal Detection Theory (SDT) and Delta Plot Analysis for Response Time Data
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

# Mapping dictionaries for categorical variables
# These convert categorical labels to numeric codes for analysis
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

# Descriptive names for each experimental condition
CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}

# Percentiles used for delta plot analysis
PERCENTILES = [10, 30, 50, 70, 90]

def read_data(file_path, prepare_for='sdt', display=False):
    """Read and preprocess data from a CSV file into SDT format.
    
    Args:
        file_path: Path to the CSV file containing raw response data
        prepare_for: Type of analysis to prepare data for ('sdt', 'delta plots', or 'raw')
        display: Whether to print summary statistics
        
    Returns:
        DataFrame with processed data in the requested format
    """
    # Read and preprocess data
    data = pd.read_csv(file_path)
    
    # Convert categorical variables to numeric codes
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    
    # Create participant number and condition index
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)
    
    if display:
        print("\nRaw data sample:")
        print(data.head())
        print("\nUnique conditions:", data['condition'].unique())
        print("Signal values:", data['signal'].unique())
    
    # Return raw data if requested
    if prepare_for == 'raw':
        return data
    
    # Transform to SDT format if requested
    if prepare_for == 'sdt':
        # Group data by participant, condition, and signal presence
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        
        if display:
            print("\nGrouped data:")
            print(grouped.head())
        
        # Transform into SDT format (hits, misses, false alarms, correct rejections)
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                
                # Get signal and noise trials
                signal_trials = c_data[c_data['signal'] == 0]
                noise_trials = c_data[c_data['signal'] == 1]
                
                if not signal_trials.empty and not noise_trials.empty:
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'hits': signal_trials['correct'].iloc[0],
                        'misses': signal_trials['nTrials'].iloc[0] - signal_trials['correct'].iloc[0],
                        'false_alarms': noise_trials['nTrials'].iloc[0] - noise_trials['correct'].iloc[0],
                        'correct_rejections': noise_trials['correct'].iloc[0],
                        'nSignal': signal_trials['nTrials'].iloc[0],
                        'nNoise': noise_trials['nTrials'].iloc[0]
                    })
        
        data = pd.DataFrame(sdt_data)
        
        if display:
            print("\nSDT summary:")
            print(data)
            if data.empty:
                print("\nWARNING: Empty SDT summary generated!")
                print("Number of participants:", len(data['pnum'].unique()))
                print("Number of conditions:", len(data['condition'].unique()))
            else:
                print("\nSummary statistics:")
                print(data.groupby('condition').agg({
                    'hits': 'sum',
                    'misses': 'sum',
                    'false_alarms': 'sum',
                    'correct_rejections': 'sum',
                    'nSignal': 'sum',
                    'nNoise': 'sum'
                }).round(2))
    
    # Prepare data for delta plot analysis
    if prepare_for == 'delta plots':
        # Initialize DataFrame for delta plot data
        dp_data = pd.DataFrame(columns=['pnum', 'condition', 'mode', 
                                      *[f'p{p}' for p in PERCENTILES]])
        
        # Process data for each participant and condition
        for pnum in data['pnum'].unique():
            for condition in data['condition'].unique():
                # Get data for this participant and condition
                c_data = data[(data['pnum'] == pnum) & (data['condition'] == condition)]
                
                # Calculate percentiles for overall RTs
                overall_rt = c_data['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['overall'],
                    **{f'p{p}': [np.percentile(overall_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for accurate responses
                accurate_rt = c_data[c_data['accuracy'] == 1]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['accurate'],
                    **{f'p{p}': [np.percentile(accurate_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for error responses
                error_rt = c_data[c_data['accuracy'] == 0]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['error'],
                    **{f'p{p}': [np.percentile(error_rt, p)] for p in PERCENTILES}
                })])
                
        if display:
            print("\nDelta plots data:")
            print(dp_data)
            
        data = pd.DataFrame(dp_data)

    return data


def apply_hierarchical_sdt_model(data):
    """Apply a hierarchical Signal Detection Theory model using PyMC.
    
    This function implements a Bayesian hierarchical model for SDT analysis,
    allowing for both group-level and individual-level parameter estimation.
    
    Args:
        data: DataFrame containing SDT summary statistics
        
    Returns:
        PyMC model object
    """
    # Get unique participants and conditions
    P = len(data['pnum'].unique())
    C = len(data['condition'].unique())
    
    # Define the hierarchical model
    with pm.Model() as sdt_model:
        # Group-level parameters
        mean_d_prime = pm.Normal('mean_d_prime', mu=0.0, sigma=1.0, shape=C)
        stdev_d_prime = pm.HalfNormal('stdev_d_prime', sigma=1.0)
        
        mean_criterion = pm.Normal('mean_criterion', mu=0.0, sigma=1.0, shape=C)
        stdev_criterion = pm.HalfNormal('stdev_criterion', sigma=1.0)
        
        # Individual-level parameters
        d_prime = pm.Normal('d_prime', mu=mean_d_prime, sigma=stdev_d_prime, shape=(P, C))
        criterion = pm.Normal('criterion', mu=mean_criterion, sigma=stdev_criterion, shape=(P, C))
        
        # Calculate hit and false alarm rates using SDT
        hit_rate = pm.math.invlogit(d_prime - criterion)
        false_alarm_rate = pm.math.invlogit(-criterion)
                
        # Likelihood for signal trials
        # Note: pnum is 1-indexed in the data, but needs to be 0-indexed for the model, so we change the indexing here.  The results table will show participant numbers starting from 0, so we need to interpret the results accordingly.
        pm.Binomial('hit_obs', 
                   n=data['nSignal'], 
                   p=hit_rate[data['pnum']-1, data['condition']], 
                   observed=data['hits'])
        
        # Likelihood for noise trials
        pm.Binomial('false_alarm_obs', 
                   n=data['nNoise'], 
                   p=false_alarm_rate[data['pnum']-1, data['condition']], 
                   observed=data['false_alarms'])
    
    return sdt_model

def draw_delta_plots(data, pnum):
    """Draw delta plots comparing RT distributions between condition pairs.
    
    Creates a matrix of delta plots where:
    - Upper triangle shows overall RT distribution differences
    - Lower triangle shows RT differences split by correct/error responses
    
    Args:
        data: DataFrame with RT percentile data
        pnum: Participant number to plot
    """
    # Filter data for specified participant
    data = data[data['pnum'] == pnum]
    
    # Get unique conditions and create subplot matrix
    conditions = data['condition'].unique()
    n_conditions = len(conditions)
    
    # Create figure with subplots matrix
    fig, axes = plt.subplots(n_conditions, n_conditions, 
                            figsize=(4*n_conditions, 4*n_conditions))
    
    # Create output directory
    OUTPUT_DIR = Path(__file__).parent / 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define marker style for plots
    marker_style = {
        'marker': 'o',
        'markersize': 10,
        'markerfacecolor': 'white',
        'markeredgewidth': 2,
        'linewidth': 3
    }
    
    # Create delta plots for each condition pair
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            # Add labels only to edge subplots
            if j == 0:
                axes[i,j].set_ylabel('Difference in RT (s)', fontsize=12)
            if i == len(axes)-1:
                axes[i,j].set_xlabel('Percentile', fontsize=12)
                
            # Skip diagonal and lower triangle for overall plots
            if i > j:
                continue
            if i == j:
                axes[i,j].axis('off')
                continue
            
            # Create masks for condition and plotting mode
            cmask1 = data['condition'] == cond1
            cmask2 = data['condition'] == cond2
            overall_mask = data['mode'] == 'overall'
            error_mask = data['mode'] == 'error'
            accurate_mask = data['mode'] == 'accurate'
            
            # Calculate RT differences for overall performance
            quantiles1 = [data[cmask1 & overall_mask][f'p{p}'].values[0] for p in PERCENTILES]
            quantiles2 = [data[cmask2 & overall_mask][f'p{p}'].values[0] for p in PERCENTILES]
            overall_delta = np.array(quantiles2) - np.array(quantiles1)
            
            # Calculate RT differences for error responses
            error_quantiles1 = [data[cmask1 & error_mask][f'p{p}'].values[0] for p in PERCENTILES]
            error_quantiles2 = [data[cmask2 & error_mask][f'p{p}'].values[0] for p in PERCENTILES]
            error_delta = np.array(error_quantiles2) - np.array(error_quantiles1)
            
            # Calculate RT differences for accurate responses
            accurate_quantiles1 = [data[cmask1 & accurate_mask][f'p{p}'].values[0] for p in PERCENTILES]
            accurate_quantiles2 = [data[cmask2 & accurate_mask][f'p{p}'].values[0] for p in PERCENTILES]
            accurate_delta = np.array(accurate_quantiles2) - np.array(accurate_quantiles1)
            
            # Plot overall RT differences
            axes[i,j].plot(PERCENTILES, overall_delta, color='black', **marker_style)
            
            # Plot error and accurate RT differences
            axes[j,i].plot(PERCENTILES, error_delta, color='red', **marker_style)
            axes[j,i].plot(PERCENTILES, accurate_delta, color='green', **marker_style)
            axes[j,i].legend(['Error', 'Accurate'], loc='upper left')

            # Set y-axis limits and add reference line
            axes[i,j].set_ylim(bottom=-1/3, top=1/2)
            axes[j,i].set_ylim(bottom=-1/3, top=1/2)
            axes[i,j].axhline(y=0, color='gray', linestyle='--', alpha=0.5) 
            axes[j,i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add condition labels
            axes[i,j].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            
            axes[j,i].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            
            plt.tight_layout()
            
    # Save the figure
    plt.savefig(OUTPUT_DIR / f'delta_plots_{pnum}.png')

# Main execution
if __name__ == "__main__":
    file_to_print = Path(__file__).parent / 'README.md'
    if file_to_print.exists():
        with open(file_to_print, 'r') as file:
            print(file.read())
    
    print("\n" + "="*80)
    print("SIGNAL DETECTION THEORY AND DELTA PLOT ANALYSIS")
    print("="*80)
    
    # Create output directory
    OUTPUT_DIR = Path(__file__).parent / 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and preprocess data
    print("\n1. LOADING AND PREPROCESSING DATA")
    print("-" * 40)
    
    data_file = Path(__file__).parent / 'data.csv'
    raw_data = read_data(data_file, prepare_for='raw', display=True)
    
    # Prepare data for SDT analysis
    sdt_data = read_data(data_file, prepare_for='sdt', display=True)
    
    # Prepare data for delta plots
    delta_data = read_data(data_file, prepare_for='delta plots', display=True)
    
    print("\n2. DESCRIPTIVE STATISTICS")
    print("-" * 40)
    
    # Calculate basic descriptive statistics
    print("\nOverall performance by condition:")
    raw_summary = raw_data.groupby(['condition', 'difficulty', 'stimulus_type']).agg({
        'accuracy': ['mean', 'std', 'count'],
        'rt': ['mean', 'std']
    }).round(3)
    print(raw_summary)
    
    print("\nSDT summary by condition:")
    sdt_summary = sdt_data.groupby('condition').agg({
        'hits': 'sum',
        'misses': 'sum', 
        'false_alarms': 'sum',
        'correct_rejections': 'sum'
    })
    
    # Calculate hit rates and false alarm rates
    sdt_summary['hit_rate'] = sdt_summary['hits'] / (sdt_summary['hits'] + sdt_summary['misses'])
    sdt_summary['fa_rate'] = sdt_summary['false_alarms'] / (sdt_summary['false_alarms'] + sdt_summary['correct_rejections'])
    sdt_summary['d_prime_basic'] = (
        np.sqrt(2) * (
            np.sqrt(np.log(1/sdt_summary['fa_rate']**2)) - 
            np.sqrt(np.log(1/sdt_summary['hit_rate']**2))
        )
    )
    print(sdt_summary.round(3))
    
    print("\n3. HIERARCHICAL SDT MODEL")
    print("-" * 40)
    
    # Build and fit the hierarchical SDT model
    print("Building hierarchical SDT model...")
    sdt_model = apply_hierarchical_sdt_model(sdt_data)
    
    # Sample from the model
    print("Sampling from model (this may take a few minutes)...")
    with sdt_model:
        # Use more conservative sampling parameters for better convergence
        trace = pm.sample(2000, tune=1000, chains=4, cores=1, 
                         target_accept=0.9, random_seed=42)
    
    print("\n4. MODEL CONVERGENCE DIAGNOSTICS")
    print("-" * 40)
    
    # Check convergence using R-hat statistics
    summary = az.summary(trace, var_names=['mean_d_prime', 'mean_criterion'])
    print("Convergence diagnostics for group-level parameters:")
    print(summary)
    
    # Check effective sample size
    print(f"\nEffective sample sizes:")
    print(f"mean_d_prime: {az.ess(trace, var_names=['mean_d_prime'])}")
    print(f"mean_criterion: {az.ess(trace, var_names=['mean_criterion'])}")
    
    # Save convergence diagnostics plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    az.plot_trace(trace, var_names=['mean_d_prime', 'mean_criterion'], axes=axes)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'convergence_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Convergence diagnostics plot saved to: {OUTPUT_DIR / 'convergence_diagnostics.png'}")
    
    print("\n5. POSTERIOR PARAMETER DISTRIBUTIONS")
    print("-" * 40)
    
    # Extract posterior samples for analysis
    mean_d_prime_samples = trace.posterior['mean_d_prime'].values
    mean_criterion_samples = trace.posterior['mean_criterion'].values
    
    # Calculate posterior means and credible intervals
    print("Group-level d' estimates (95% CI):")
    for i, condition in enumerate(CONDITION_NAMES.keys()):
        d_prime_mean = np.mean(mean_d_prime_samples[:, :, i])
        d_prime_ci = np.percentile(mean_d_prime_samples[:, :, i], [2.5, 97.5])
        print(f"  {CONDITION_NAMES[condition]}: {d_prime_mean:.3f} [{d_prime_ci[0]:.3f}, {d_prime_ci[1]:.3f}]")
    
    print("\nGroup-level criterion estimates (95% CI):")
    for i, condition in enumerate(CONDITION_NAMES.keys()):
        criterion_mean = np.mean(mean_criterion_samples[:, :, i])
        criterion_ci = np.percentile(mean_criterion_samples[:, :, i], [2.5, 97.5])
        print(f"  {CONDITION_NAMES[condition]}: {criterion_mean:.3f} [{criterion_ci[0]:.3f}, {criterion_ci[1]:.3f}]")
    
    # Create posterior distribution plots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Plot d-prime distributions
    for i, condition in enumerate(CONDITION_NAMES.keys()):
        axes[0, i].hist(mean_d_prime_samples[:, :, i].flatten(), bins=50, alpha=0.7, density=True)
        axes[0, i].set_title(f"d' - {CONDITION_NAMES[condition]}")
        axes[0, i].set_xlabel("d-prime")
        axes[0, i].set_ylabel("Density")
    
    # Plot criterion distributions  
    for i, condition in enumerate(CONDITION_NAMES.keys()):
        axes[1, i].hist(mean_criterion_samples[:, :, i].flatten(), bins=50, alpha=0.7, density=True)
        axes[1, i].set_title(f"Criterion - {CONDITION_NAMES[condition]}")
        axes[1, i].set_xlabel("Criterion")
        axes[1, i].set_ylabel("Density")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'posterior_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Posterior distributions plot saved to: {OUTPUT_DIR / 'posterior_distributions.png'}")
    
    print("\n6. EFFECT SIZE ANALYSIS")
    print("-" * 40)
    
    # Calculate effect sizes for stimulus type and difficulty
    # Difficulty effect on d-prime (Hard - Easy)
    difficulty_effect_dprime = (
        (mean_d_prime_samples[:, :, 2] + mean_d_prime_samples[:, :, 3]) / 2 -  # Hard conditions
        (mean_d_prime_samples[:, :, 0] + mean_d_prime_samples[:, :, 1]) / 2    # Easy conditions
    )
    
    # Stimulus type effect on d-prime (Complex - Simple) 
    stimulus_effect_dprime = (
        (mean_d_prime_samples[:, :, 1] + mean_d_prime_samples[:, :, 3]) / 2 -  # Complex conditions
        (mean_d_prime_samples[:, :, 0] + mean_d_prime_samples[:, :, 2]) / 2    # Simple conditions
    )
    
    # Difficulty effect on criterion
    difficulty_effect_criterion = (
        (mean_criterion_samples[:, :, 2] + mean_criterion_samples[:, :, 3]) / 2 -
        (mean_criterion_samples[:, :, 0] + mean_criterion_samples[:, :, 1]) / 2
    )
    
    # Stimulus type effect on criterion
    stimulus_effect_criterion = (
        (mean_criterion_samples[:, :, 1] + mean_criterion_samples[:, :, 3]) / 2 -
        (mean_criterion_samples[:, :, 0] + mean_criterion_samples[:, :, 2]) / 2
    )
    
    print("Effect sizes (posterior mean and 95% CI):")
    print(f"Difficulty effect on d': {np.mean(difficulty_effect_dprime):.3f} [{np.percentile(difficulty_effect_dprime, 2.5):.3f}, {np.percentile(difficulty_effect_dprime, 97.5):.3f}]")
    print(f"Stimulus type effect on d': {np.mean(stimulus_effect_dprime):.3f} [{np.percentile(stimulus_effect_dprime, 2.5):.3f}, {np.percentile(stimulus_effect_dprime, 97.5):.3f}]")
    print(f"Difficulty effect on criterion: {np.mean(difficulty_effect_criterion):.3f} [{np.percentile(difficulty_effect_criterion, 2.5):.3f}, {np.percentile(difficulty_effect_criterion, 97.5):.3f}]")
    print(f"Stimulus type effect on criterion: {np.mean(stimulus_effect_criterion):.3f} [{np.percentile(stimulus_effect_criterion, 2.5):.3f}, {np.percentile(stimulus_effect_criterion, 97.5):.3f}]")
    
    # Plot effect size distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].hist(difficulty_effect_dprime.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].axvline(0, color='red', linestyle='--')
    axes[0, 0].set_title("Difficulty Effect on d'")
    axes[0, 0].set_xlabel("Effect Size")
    
    axes[0, 1].hist(stimulus_effect_dprime.flatten(), bins=50, alpha=0.7, color='green')
    axes[0, 1].axvline(0, color='red', linestyle='--')
    axes[0, 1].set_title("Stimulus Type Effect on d'")
    axes[0, 1].set_xlabel("Effect Size")
    
    axes[1, 0].hist(difficulty_effect_criterion.flatten(), bins=50, alpha=0.7, color='blue')
    axes[1, 0].axvline(0, color='red', linestyle='--')
    axes[1, 0].set_title("Difficulty Effect on Criterion")
    axes[1, 0].set_xlabel("Effect Size")
    
    axes[1, 1].hist(stimulus_effect_criterion.flatten(), bins=50, alpha=0.7, color='green')
    axes[1, 1].axvline(0, color='red', linestyle='--')
    axes[1, 1].set_title("Stimulus Type Effect on Criterion")
    axes[1, 1].set_xlabel("Effect Size")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'effect_sizes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Effect sizes plot saved to: {OUTPUT_DIR / 'effect_sizes.png'}")
    
    print("\n7. DELTA PLOTS ANALYSIS")
    print("-" * 40)
    
    # Generate delta plots for each participant
    participants = delta_data['pnum'].unique()
    print(f"Generating delta plots for {len(participants)} participants...")
    
    for pnum in participants[:3]:  # Show first 3 participants as examples
        print(f"Creating delta plots for participant {pnum}...")
        draw_delta_plots(delta_data, pnum)
    
    print(f"Delta plots saved to: {OUTPUT_DIR}")
    
    print("\n8. SUMMARY AND INTERPRETATION")
    print("-" * 40)
    
    # Calculate probability of meaningful effects
    prob_difficulty_negative = np.mean(difficulty_effect_dprime < -0.1)
    prob_stimulus_negative = np.mean(stimulus_effect_dprime < -0.1)
    
    print("\nSIGNAL DETECTION THEORY RESULTS:")
    print(f"• Difficulty manipulation decreases d' by {abs(np.mean(difficulty_effect_dprime)):.3f} on average")
    print(f"• Probability that difficulty effect on d' is meaningful (< -0.1): {prob_difficulty_negative:.3f}")
    print(f"• Stimulus type manipulation decreases d' by {abs(np.mean(stimulus_effect_dprime)):.3f} on average")
    print(f"• Probability that stimulus type effect on d' is meaningful (< -0.1): {prob_stimulus_negative:.3f}")
    
    if abs(np.mean(difficulty_effect_dprime)) > abs(np.mean(stimulus_effect_dprime)):
        print(f"• Difficulty manipulation has a LARGER effect on discriminability than stimulus type")
    else:
        print(f"• Stimulus type manipulation has a LARGER effect on discriminability than difficulty")
    
    print(f"\nCRITERION EFFECTS:")
    print(f"• Difficulty effect on criterion: {np.mean(difficulty_effect_criterion):.3f}")
    print(f"• Stimulus type effect on criterion: {np.mean(stimulus_effect_criterion):.3f}")
    
    print(f"\nDELTA PLOTS INTERPRETATION:")
    print(f"• Delta plots show RT distribution differences between conditions")
    print(f"• Upper triangle: Overall RT differences")
    print(f"• Lower triangle: Error (red) vs Accurate (green) RT differences")
    print(f"• Patterns suggest diffusion model parameter changes")
    
    print(f"\nCOMPARISON OF MANIPULATIONS:")
    if abs(np.mean(difficulty_effect_dprime)) > abs(np.mean(stimulus_effect_dprime)) * 1.5:
        print(f"• Difficulty manipulation has a SUBSTANTIALLY larger effect than stimulus type")
    elif abs(np.mean(stimulus_effect_dprime)) > abs(np.mean(difficulty_effect_dprime)) * 1.5:
        print(f"• Stimulus type manipulation has a SUBSTANTIALLY larger effect than difficulty")
    else:
        print(f"• Both manipulations have similar magnitude effects on performance")
    
    print(f"\nOUTPUT FILES CREATED:")
    print(f"• {OUTPUT_DIR / 'convergence_diagnostics.png'}")
    print(f"• {OUTPUT_DIR / 'posterior_distributions.png'}")
    print(f"• {OUTPUT_DIR / 'effect_sizes.png'}")
    print(f"• {OUTPUT_DIR / 'delta_plots_*.png'}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)