from setup import *  
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, chi2
from scipy.special import erf, erfinv

def generate_sample_data():
    """Generate hypothetical data for one participant across K=3 conditions
    
    Returns:
    --------
    dict
        Dictionary containing the generated data
    """
    # Data for one participant across K=3 conditions
    conditions = ['Easy', 'Medium', 'Hard']
    K = len(conditions)

    # Hits for each condition
    hits_k = np.array([90, 75, 60])
    # Misses for each condition
    misses_k = np.array([10, 25, 40])
    # False Alarms for each condition
    fas_k = np.array([10, 20, 25])
    # Correct Rejections for each condition
    crs_k = np.array([90, 80, 75])

    # Calculate trials per condition
    n_signal_trials_k = hits_k + misses_k
    n_noise_trials_k = fas_k + crs_k

    # Create difficulty predictor: -1 for Easy, 0 for Medium, 1 for Hard
    difficulty = np.array([-1, 0, 1])

    print(f"Conditions: {conditions}")
    print(f"Difficulty values: {difficulty}")
    print(f"N Signal Trials per condition: {n_signal_trials_k}")
    print(f"N Noise Trials per condition: {n_noise_trials_k}")
    
    return {
        'conditions': conditions,
        'K': K,
        'hits_k': hits_k,
        'misses_k': misses_k,
        'fas_k': fas_k,
        'crs_k': crs_k,
        'n_signal_trials_k': n_signal_trials_k,
        'n_noise_trials_k': n_noise_trials_k,
        'difficulty': difficulty
    }

def build_hierarchical_model(data):
    """Build the hierarchical SDT model with linear d' and common criterion
    
    Parameters:
    -----------
    data : dict
        Dictionary containing the observed data
        
    Returns:
    --------
    pm.Model
        PyMC model for hierarchical SDT
    """
    coords = {"condition": data['conditions']}

    with pm.Model(coords=coords) as model_hierarchical:
        # --- Priors ---
        # Baseline d' (at medium difficulty)
        d_prime_baseline = pm.Normal('d_prime_baseline', mu=0.0, sigma=2.0)
        
        # Slope for d' change with difficulty
        d_prime_slope = pm.Normal('d_prime_slope', mu=0.0, sigma=1.0)
        
        # Single criterion across all conditions
        criterion = pm.Normal('criterion', mu=0.0, sigma=0.5)
        
        # --- Linear model for d' ---
        # d' = baseline + slope * difficulty
        # This creates a vector of d' values following the linear relationship
        d_prime = pm.Deterministic('d_prime', 
                                  d_prime_baseline + d_prime_slope * data['difficulty'],
                                  dims="condition")

        # --- Deterministic Transformations ---
        # These calculations are vectorized over the conditions
        # Note that criterion is now a scalar, not a vector
        hr_D = pm.Deterministic('hr_D', Phi(d_prime / 2 - criterion), dims="condition")
        far_D = pm.Deterministic('far_D', Phi(-d_prime / 2 - criterion), dims="condition")

        # --- Likelihood ---
        H_obs = pm.Binomial('H_obs',
                          n=data['n_signal_trials_k'],
                          p=hr_D,
                          observed=data['hits_k'],
                          dims="condition")

        FA_obs = pm.Binomial('FA_obs',
                           n=data['n_noise_trials_k'],
                           p=far_D,
                           observed=data['fas_k'],
                           dims="condition")
                           
    return model_hierarchical

def sample_posterior(model, draws, tune, chains, target_accept):
    """Sample from the posterior distribution
    
    Parameters:
    -----------
    model : pm.Model
        The PyMC model to sample from
    draws : int
        Number of samples per chain after tuning
    tune : int
        Number of steps to discard for tuning the sampler
    chains : int
        Number of independent chains to run
    target_accept : float
        Parameter for NUTS algorithm, higher values can help with difficult posteriors
        
    Returns:
    --------
    az.InferenceData
        Posterior samples
    """
    with model:
        # Draw samples from the posterior
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept)
    return idata

def analyze_results(idata):
    """Analyze and visualize the posterior samples
    
    Parameters:
    -----------
    idata : az.InferenceData
        Posterior samples
    """
    # Check summary statistics
    print("Hierarchical SDT Model Summary:")
    # Get summary statistics with HDI intervals 
    summary = az.summary(idata, 
                         var_names=['d_prime', 'd_prime_baseline', 'd_prime_slope', 'criterion'],
                         hdi_prob=0.94)
    print(summary)
    
    # Check trace plots
    az.plot_trace(idata, var_names=['d_prime_baseline', 'd_prime_slope', 'criterion'])
    plt.tight_layout()
    plt.show()
    
    # Plot posterior distributions
    az.plot_posterior(idata, var_names=['d_prime_baseline', 'd_prime_slope', 'criterion'], 
                     hdi_prob=0.94, ref_val=None)
    plt.tight_layout()
    plt.show()
    
    # Plot posterior for d' across conditions
    pm.plot_forest(idata, var_names=['d_prime'], 
                  combined=True, hdi_prob=0.94)
    plt.title("d' by condition (linearly constrained)")
    plt.tight_layout()
    plt.show()

def run_hierarchical_analysis():
    """Run the complete analysis for the hierarchical SDT model"""
    print_version_info()
    data = generate_sample_data()
    model = build_hierarchical_model(data)
    idata = sample_posterior(model, draws=2000, tune=1000, chains=4, target_accept=0.9)
    analyze_results(idata)
    return model, idata

if __name__ == "__main__":
    run_hierarchical_analysis()