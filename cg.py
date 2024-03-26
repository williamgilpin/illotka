import numpy as np

class ConjugateGradientMethod:
    """
    Solve a linear system Ax = b using the conjugate gradient method. The input matrix
    must be a symmetric, positive definite matrix.

    Parameters
        tol (float): Error tolerance to use as stopping criteria.
        max_iter (int): Maximum number of iterations to perform.
        store_history (bool): If True, store the history of the solution at each 
            iteration.
    """

    def __init__(self, max_iter=1000, store_history=False, tol=1e-6):
        self.max_iter = max_iter
        self.store_history = store_history
        self.tol = tol

    def solve(self, A, b):

        if self.store_history:
            self.history = list()

        x = np.zeros_like(b)
        r = b
        p = r

        for i in range(self.max_iter):
            alpha = (r.T @ r) / (p.T @ A @ p)
            x = x + alpha * p
            r = r - alpha * A @ p

            if np.linalg.norm(r) < self.tol:
                break

            beta = (r.T @ r) / (b.T @ b)
            p = r + beta * p

            if self.store_history:
                self.history.append(x.copy())

        return x