"""
Signal Detection Theory (SDT) Models - Common Setup

This module provides the common imports and helper functions needed by the SDT models.
"""
import pymc as pm
import arviz as az
import numpy as np
import scipy as scipy
import scipy.stats as stats
import pytensor.tensor as pt  # For wrapping scipy functions if needed, or use pm.math
import sys

# For plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Helper function for standard normal CDF (probit function inverse)
def Phi(x):
    """Standard normal CDF
    
    PyMC provides pm.math.invprobit which is exactly this function.
    
    Parameters:
    -----------
    x : float or array-like
        Input values
        
    Returns:
    --------
    float or array-like
        Cumulative probability according to the standard normal distribution
    """
    # Alternative implementation: 
    # return 0.5 + 0.5 * pt.erf(x / pt.sqrt(2.))
    return pm.math.invprobit(x)  # Use PyMC's built-in function

def print_version_info():
    """Print version information for PyMC and ArviZ"""
    print(f"Running python {sys.version}")
    print(f"Running on PyMC v{pm.__version__}")
    print(f"Running on ArviZ v{az.__version__}")
    print(f"Running on numpy v{np.__version__}")
    print(f"Running on scipy v{scipy.__version__}")
    print(f"Running on matplotlib v{mpl.__version__}")
    # print(f"Running on seaborn v{sns.__version__}")

if __name__ == "__main__":
    print_version_info()