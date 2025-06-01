
import numpy as np


from scipy.stats import truncnorm
def normal_truncated(size, mu=0, sigma=1, clip=1):
    """
    Sample from a truncated normal distribution.

    Args:
        size: The number of samples to draw.
        mu: The mean of the normal distribution.
        sigma: The standard deviation of the normal distribution.
        clip: The truncation point.

    Returns:
        The samples from the truncated normal distribution.
    """
    a, b = -clip, clip
    samples = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size)
    samples /= np.std(samples)
    return samples

from scipy.stats import gennorm
def normal_generalized(size, mu=0, sigma=1, beta=0.1):
    """
    Sample from a generalized normal distribution.

    Args:
        size: The number of samples to draw.
        mu: The mean of the distribution.
        sigma: The standard deviation of the distribution.
        beta: The shape parameter of the distribution.

    Returns:
        The samples from the generalized normal distribution.
    """
    samples = gennorm.rvs(beta, loc=mu, scale=sigma, size=size)
    samples /= np.std(samples)
    return samples



def levenshtein(a, b):
    "Calculates the Levenshtein distance between two lists a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n
        
    current_row = range(n + 1)
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous_row[j] + 1, current_row[j - 1] + 1
            change = previous_row[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)
            
    return current_row[n]


def num_permutations(arr):
    """Given a 2D array, detect changes in the sort order along the second axis."""
    ordering = np.argsort(arr, axis=1)
    delta = np.diff(ordering, axis=0)
    swap_times = np.abs(np.sum(np.abs(delta), axis=1)) > 0
    return np.sum(swap_times)
    
def detect_zero_crossings(traj):
    """Detect zero crossings in a trajectory."""
    return np.where(np.diff(np.sign(traj), axis=0))[0]

def bilinear_interpolation(xa, xb, ya, yb, s, t):
    P1 = s * xa + (1 - s) * xb
    P2 = s * ya + (1 - s) * yb
    return t * P1 + (1 - t) * P2

def progress_bar(i, n, n_bar=20):
    """
    Print a progress bar to stdout

    Args:
        i (int): Current iteration
        n (int): Total number of iterations
        n_bar (int): Number of characters in the progress bar

    Returns:
        None
    """
    idots = int(i / n * n_bar)
    stars = '#' * idots
    spaces = ' ' * (n_bar - idots)
    bar_str = f"[{stars}{spaces}] "
    print(bar_str, end='\r')
    if i == n - 1:
        print("\n")
    return None


def bilinear_interpolation(xa, xb, ya, yb, s, t):
    """
    Perform bilinear interpolation on a grid of points.

    Args:
        xa (np.ndarray): x-coordinates of the lower left corner of the grid
        xb (np.ndarray): x-coordinates of the lower right corner of the grid
        ya (np.ndarray): y-coordinates of the lower left corner of the grid
        yb (np.ndarray): y-coordinates of the upper left corner of the grid
        s (float): Value between 0 and 1 for the x-coordinate
        t (float): Value between 0 and 1 for the y-coordinate

    Returns:
        The interpolated value
    """
    P1 = s * xa + (1 - s) * xb
    P2 = s * ya + (1 - s) * yb
    return t * P1 + (1 - t) * P2


def compute_settling_time(traj, tvals=None, eps=1e-6):
    """
    Compute the settling time from a multivariate time series. 
    If the trajectory does not reach the steady-state within the integration time, 
    return a NaN.
    """
    final_value = traj[-1]
    scaled_distance = (np.linalg.norm(traj - final_value, axis=1)) / np.linalg.norm(traj, axis=1)
    final_index = np.where(scaled_distance < eps)[0]
    if len(final_index) == 0:
        return np.nan
    else:
        return final_index[0]