"""
Model 2: Conditional SDT (One Person, K Conditions)

This model extends the basic SDT model to handle data from one person across
multiple experimental conditions (e.g., different stimulus difficulties, 
different payoffs). We expect d' or c (or both) might vary across conditions.
"""
from setup import *  # Import common functions and libraries

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

    print(f"Conditions: {conditions}")
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
        'n_noise_trials_k': n_noise_trials_k
    }

def build_model(data):
    """Build the conditional SDT model
    
    Parameters:
    -----------
    data : dict
        Dictionary containing the observed data
        
    Returns:
    --------
    pm.Model
        PyMC model for conditional SDT
    """
    coords = {"condition": data['conditions']}

    with pm.Model(coords=coords) as model_conditional:
        # --- Priors ---
        # We now estimate K values for d' and c, one for each condition.
        # The priors are the same type as before, but now define K independent parameters.
        # Use the `dims` argument to link the parameters to the coordinate names.
        d_prime = pm.Normal('d_prime', mu=0.0, sigma=2.0, dims="condition")
        criterion = pm.Normal('criterion', mu=0.0, sigma=0.5, dims="condition")

        # --- Deterministic Transformations ---
        # These calculations are now vectorized over the conditions.
        hr_D = pm.Deterministic('hr_D', Phi(d_prime / 2 - criterion), dims="condition")
        far_D = pm.Deterministic('far_D', Phi(-d_prime / 2 - criterion), dims="condition")

        # --- Likelihood ---
        # The observed counts are also vectors, one element per condition.
        # The likelihood is applied element-wise.
        H_obs = pm.Binomial('H_obs',
                          n=data['n_signal_trials_k'], # Vector of N_signal trials
                          p=hr_D,                    # Vector of hit rates
                          observed=data['hits_k'],    # Vector of observed hits
                          dims="condition")

        FA_obs = pm.Binomial('FA_obs',
                           n=data['n_noise_trials_k'], # Vector of N_noise trials
                           p=far_D,                   # Vector of false alarm rates
                           observed=data['fas_k'],     # Vector of observed false alarms
                           dims="condition")
                           
    return model_conditional

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
    print("Conditional SDT Model Summary:")
    # Get summary statistics with HDI intervals 
    summary_conditional = az.summary(idata, 
                                    var_names=['d_prime', 'criterion', 'hr_D', 'far_D'],
                                    hdi_prob=0.94)
    print(summary_conditional)
    
    # Check trace plots
    az.plot_trace(idata, var_names=['d_prime', 'criterion'])
    plt.tight_layout()
    plt.show()
    
    # Plot posterior distributions
    az.plot_posterior(idata, var_names=['d_prime', 'criterion'], 
                     hdi_prob=0.94, ref_val=None)
    plt.tight_layout()
    plt.show()
    
    # Compare conditions
    pm.plot_forest(idata, var_names=['d_prime'], 
                  combined=True, hdi_prob=0.94)
    plt.title("d' by condition")
    plt.tight_layout()
    plt.show()
    
    pm.plot_forest(idata, var_names=['criterion'], 
                  combined=True, hdi_prob=0.94)
    plt.title("criterion by condition")
    plt.tight_layout()
    plt.show()

def run_analysis():
    """Run the complete analysis for the conditional SDT model"""
    print_version_info()
    data = generate_sample_data()
    model = build_model(data)
    idata = sample_posterior(model, draws=2000, tune=1000, chains=4, target_accept=0.9)
    analyze_results(idata)
    return model, idata

if __name__ == "__main__":
    run_analysis()