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