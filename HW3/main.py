"""
MPT Model for the Weapon Identification Task

The data we use is due to Rivers (2017):

    Rivers, A. M. (2017). The weapons identification task: Recommendations
    for adequately powered research. PLoS ONE, 12(6), e0177857. 
    https://doi.org/10.1371/journal.pone.0177857
    
The data are directly fetched from the OSF repository:
    https://osf.io/5jzhr/download

The models are found in Heck et al. (2023), starting on page 572 and illustrated in Figure 3:

    Heck, D. W., et al. (2023). A review of applications of the Bayes factor in psychological research. Psychological Methods, 28(3), 558-579. 
    https://doi.org/10.1037/met0000454
    
Throughout, we will ignore the neutral priming condition.
"""

import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import sys

###############################################################################
### --- Model definitions ---
###############################################################################

# Stroop + Guessing Model
#   Parameters:
#     alpha: Probability of an automatic, stereotype-consistent response.
#     beta: Probability of controlled, accurate response.
#     gamma: Probability of guessing "tool" when guessing occurs.
#   Model Structure:
#     Sequential processing: automatic → controlled → guessing.

def fit_stroop_model(data, n_trials):
    """Fits the Stroop + Guessing MPT model and returns the trace."""

    with pm.Model() as model:
        
        alpha = pm.Uniform("alpha", 0, 1)
        beta = pm.Uniform("beta", 0, 1)
        gamma = pm.Beta("gamma", 3, 3)

        theta_wt = alpha + (1 - alpha) * beta + (1 - alpha) * (1 - beta) * gamma
        theta_bt = (1 - alpha) * beta + (1 - alpha) * (1 - beta) * gamma
        theta_wg = (1 - alpha) * beta + (1 - alpha) * (1 - beta) * (1 - gamma)
        theta_bg = alpha + (1 - alpha) * beta + (1 - alpha) * (1 - beta) * (1 - gamma)

        pm.Binomial("white_tool", n=n_trials, p=theta_wt, observed=data["white_tool"])
        pm.Binomial("black_tool", n=n_trials, p=theta_bt, observed=data["black_tool"])
        pm.Binomial("white_gun", n=n_trials, p=theta_wg, observed=data["white_gun"])
        pm.Binomial("black_gun", n=n_trials, p=theta_bg, observed=data["black_gun"])

        trace = pm.sample(1000, tune=1000, target_accept=0.9, idata_kwargs={"log_likelihood": True})

    return trace


# Process Dissociation + Guessing Model
#   Parameters:
#     A: Probability of automatic processing.
#     C: Probability of controlled processing.
#     B: Probability of guessing "tool" when guessing occurs.
#   Model Structure:
#     Automatic and controlled processing are independent branches, 
#     allowing for dual routes to correct/incorrect responses.

def fit_pd_model(data, n_trials):
    """Fits the Process Dissociation + Guessing MPT model and returns the trace."""
    pass


# Saturated MPT Model
#   Parameters:
#     theta_wt: white tool correct response
#     theta_bt: black tool correct response
#     theta_wg: white gun correct response
#     theta_bg: black gun correct response
#   Model Structure:
#     Each response type has its own independent probability of correct response.

def fit_saturated_model(data, n_trials):
    """Fits the Saturated MPT model and returns the trace."""

    with pm.Model() as model:

        theta_wt = pm.Beta("theta_wt", 1, 1)
        theta_bt = pm.Beta("theta_bt", 1, 1)
        theta_wg = pm.Beta("theta_wg", 1, 1)
        theta_bg = pm.Beta("theta_bg", 1, 1)

        pm.Binomial("white_tool", n=n_trials, p=theta_wt, observed=data["white_tool"])
        pm.Binomial("black_tool", n=n_trials, p=theta_bt, observed=data["black_tool"])
        pm.Binomial("white_gun", n=n_trials, p=theta_wg, observed=data["white_gun"])
        pm.Binomial("black_gun", n=n_trials, p=theta_bg, observed=data["black_gun"])

        trace = pm.sample(1000, tune=1000, target_accept=0.9, idata_kwargs={"log_likelihood": True})

    return trace

###############################################################################
### --- Runner function ---
###############################################################################

def run_all_models(conditions_data, num_trials):
    """Runs all model fits and returns a dictionary of traces."""

    model_traces = {}

    for deadline, data in conditions_data.items():
        print(f"\n> --- Fitting models for {deadline} ---")
        print("> Fitting Stroop model...")
        model_traces[f"stroop_{deadline}"] = fit_stroop_model(data, num_trials)
        print("> Fitting PD model...")
        model_traces[f"pd_{deadline}"] = fit_pd_model(data, num_trials)
        print("> Fitting Saturated model...")
        model_traces[f"saturated_{deadline}"] = fit_saturated_model(data, num_trials)

    return model_traces 


###############################################################################
### --- Data loading ---
###############################################################################

def load_rivers_data():
    """
    Loads the Rivers (2017) dataset directly from OSF.
    There's some poor formatting in the raw data, so we need to clean it up a little.

    Args:
        url (str): The direct download URL for the data file.

    Returns:
        dict: A dictionary with keys '500ms' and '1000ms', each containing a dictionary with keys 'white_tool', 'black_tool', 'white_gun', and 'black_gun'.
    """

    DATA_URL = "https://osf.io/5jzhr/download"

    # Data needs to be cleaned up a little
    df = pd.read_csv(DATA_URL, sep=',', skipinitialspace=True, header=0, engine='python')
    df.columns = ['deadline500ms', 'deadline1000ms']
    df.index = df.index.str.strip()
    
    print(df)

    data = {
        "500ms": {
            "white_tool": df.loc["white_tool_1", "deadline500ms"],
            "black_tool": df.loc["black_tool_1", "deadline500ms"],
            "white_gun": df.loc["white_gun_1", "deadline500ms"],
            "black_gun": df.loc["black_gun_1", "deadline500ms"]
        },
        "1000ms": {
            "white_tool": df.loc["white_tool_1", "deadline1000ms"],
            "black_tool": df.loc["black_tool_1", "deadline1000ms"],
            "white_gun": df.loc["white_gun_1", "deadline1000ms"],
            "black_gun": df.loc["black_gun_1", "deadline1000ms"]
        }
    }
    return data


###############################################################################
### --- Helper functions ---
###############################################################################

def get_var_names_from_idata(idata):
    """Helper function to get the variable names from an ArviZ IData object."""
    return [v for v in idata.posterior.variables if v not in ['chain', 'draw']]

def ensure_figure_directory_exists():
    """Ensures that the figure directory exists."""
    try:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Figures will be saved to: {FIG_DIR.resolve()}")
    except OSError as e:
        print(f"Error creating figure directory {FIG_DIR}: {e}", file=sys.stderr)


###############################################################################
### --- Postprocessing ---
###############################################################################

def make_figures(all_results, model, condition):
    """Makes all the figures for a model and condition."""
    
    if model not in ["stroop", "pd", "saturated"]:
        raise ValueError("Model must be either 'stroop', 'pd', or 'saturated'")
    
    if condition not in ["500ms", "1000ms"]:
        raise ValueError("Condition must be either '500ms' or '1000ms'")
    
    idata = all_results[f"{model}_{condition}"]
    
    if idata is None:
        print(f"Skipping figures for {model} ({condition}): Not implemented yet")
        return
    
    var_names = get_var_names_from_idata(idata)
    
    # Default colors that will cycle if there are more variables than colors
    colors = ['#0000DD', '#DD0000', '#DD9500', '#00DD00']

    az.plot_pair(
        idata,
        var_names=var_names,
        kind="kde",
        figsize=(8, 8)
    )
    plt.suptitle(f"{model} ({condition}) Posterior Pair Plot", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{model}_{condition}_pairplot.png")

    # Create Matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(8, 5))

    for var_name, color in zip(var_names, colors * ((len(var_names) // len(colors)) + 1)):
        az.plot_density(
            idata,
            var_names=[var_name],
            hdi_prob=1.0,
            shade=0.2,
            point_estimate=None,
            ax=ax,
            colors=color
        )

    ax.set_title(f"{model} ({condition}) marginal posteriors")
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{model}_{condition}_marginal_posteriors.png")

def print_summary(all_results, model, condition):
    """Prints the summary of a model for a given condition."""
    # Print summary
    idata = all_results[f"{model}_{condition}"]
    if idata is None:
        print(f"\n{model} ({condition}) model summary: Not implemented yet")
        return
        
    var_names = get_var_names_from_idata(idata)
    print(f"\n{model} ({condition}) model summary:")
    print(az.summary(idata, var_names=var_names))


###############################################################################
### --- Constants ---
###############################################################################

N_TRIALS = 1440

ROOT_DIR = Path(__file__).parent.parent
FIG_DIR = ROOT_DIR / "figures"


###############################################################################
### --- Main execution block ---
###############################################################################

if __name__ == "__main__":
    
    ensure_figure_directory_exists()

    data = load_rivers_data()

    all_results = run_all_models(data, N_TRIALS)
    
    # --- Loop over all cases ---
    for model in ["stroop", "pd", "saturated"]:
        for condition in ["500ms", "1000ms"]:
            print_summary(all_results, model, condition)
            make_figures(all_results, model, condition)