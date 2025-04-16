"""
Model 1: Basic SDT (One Person, One Condition)

This is the simplest case. We have data from one observer performing the task 
under a single condition. This model estimates sensitivity (d') and criterion (c)
parameters for a single participant.
"""
from setup import *  # Import common functions and libraries

def generate_sample_data():
    """Generate hypothetical data for one participant
    
    Returns:
    --------
    dict
        Dictionary containing the generated data
    """
    # Data for one participant
    hits = 75
    misses = 25
    false_alarms = 20
    correct_rejections = 80

    n_signal_trials = hits + misses
    n_noise_trials = false_alarms + correct_rejections

    print(f"N Signal Trials: {n_signal_trials}")
    print(f"N Noise Trials: {n_noise_trials}")
    print(f"Observed HR: {hits / n_signal_trials:.2f}")
    print(f"Observed FAR: {false_alarms / n_noise_trials:.2f}")
    
    return {
        'hits': hits,
        'misses': misses,
        'false_alarms': false_alarms,
        'correct_rejections': correct_rejections,
        'n_signal_trials': n_signal_trials,
        'n_noise_trials': n_noise_trials
    }

def build_model(data):
    """Build the basic SDT model
    
    Parameters:
    -----------
    data : dict
        Dictionary containing the observed data
        
    Returns:
    --------
    pm.Model
        PyMC model for basic SDT
    """
    # Create a model object that will contain all the variables and distributions
    with pm.Model() as model_basic:
        # --- Priors ---
        # Sensitivity (d'). Normal prior, often centered at 0.
        # Standard deviation of 2.0 and 0.5 imply uniform distributions on hit and false alarm rates.
        d_prime = pm.Normal('d_prime', mu=0.0, sigma=2.0)

        # Criterion (c). Normal prior, also often centered at 0.
        criterion = pm.Normal('criterion', mu=0.0, sigma=0.5)

        # --- Deterministic Transformations ---
        # Calculate the theoretical Hit Rate and False Alarm Rate from d' and c
        # Using our Phi helper function (which uses pm.math.invprobit)
        hr_D = pm.Deterministic('hr_D', Phi(d_prime / 2 - criterion))
        far_D = pm.Deterministic('far_D', Phi(-d_prime / 2 - criterion))

        # --- Likelihood ---
        # The observed counts (Hits, False Alarms) follow Binomial distributions
        # conditioned on the number of trials and the theoretical rates (hr_D, far_D).

        # Likelihood for Hits (given signal trials)
        H_obs = pm.Binomial('H_obs',
                          n=data['n_signal_trials'],
                          p=hr_D,
                          observed=data['hits'])

        # Likelihood for False Alarms (given noise trials)
        FA_obs = pm.Binomial('FA_obs',
                           n=data['n_noise_trials'],
                           p=far_D,
                           observed=data['false_alarms'])
                           
    return model_basic

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
    print("Basic SDT Model Summary:")
    summary_basic = az.summary(idata, var_names=['d_prime', 'criterion', 'hr_D', 'far_D'], 
                               hdi_prob=0.94)
    print(summary_basic)

    # Check trace plots for convergence diagnostics
    az.plot_trace(idata, var_names=['d_prime', 'criterion'])
    plt.tight_layout()
    plt.show()

    # Plot posterior distributions
    az.plot_posterior(idata, var_names=['d_prime', 'criterion'], 
                     hdi_prob=0.94, ref_val=None)
    plt.tight_layout()
    plt.show()

def run_analysis():
    """Run the complete analysis for the basic SDT model"""
    print_version_info()
    data = generate_sample_data()
    model = build_model(data)
    idata = sample_posterior(model, draws=2000, tune=1000, chains=4, target_accept=0.9)
    analyze_results(idata)
    return model, idata

if __name__ == "__main__":
    run_analysis()

