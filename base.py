from scipy.integrate import solve_ivp
import numpy as np


class RandomLotkaVolterra:
    """
    Generate random foodwebs using the model of Serván et al. (2018).

    Parameters:
        n (int): number of species
        sigma (float): standard deviation of interaction strengths
        kfrac (float): fraction of species that are swapped
        eps (float): strength of perturbation
        connectivity (float): fraction of nonzero interactions
        d (float): self-interaction strength
        random_state (int): random seed
        n_max (int): maximum number of species to generate. Used to control the 
            random seed and to truncate the interaction matrix.
        early_stopping (bool): stop the integration when the time derivative is small
        verbose (bool): print out stability and rank of the interaction matrix
        tolerance (float): tolerance for stopping the integration

    References:
        Serván et al. (2018).
    """
    def __init__(self, n, sigma=1.0, kfrac=1/200, eps=0.0, connectivity=1.0, d=1.0, random_state=0, n_max=1000, 
                 early_stopping=True, verbose=False, tolerance=1e-10):
        self.n = n
        self.d = d
        self.sigma = sigma
        self.connectivity = connectivity
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.tolerance = tolerance

        self.kfrac = kfrac
        self.eps = eps

        np.random.seed(random_state)
        self.random_state = random_state

        self.A = np.random.normal(size=(n_max, n_max), scale=self.sigma)
        connectivity_mask = np.random.uniform(size=(n_max, n_max)) < self.connectivity
        self.A = self.A * connectivity_mask
        self.r = np.random.normal(size=n_max)
        self.dvals = np.random.normal(size=n_max) - self.d
        np.fill_diagonal(self.A, self.dvals)
        # np.fill_diagonal(self.A, 0)

        

        # self.dvals = self.dvals[:n]
        self.A = self.A[:n, :n]
        self.r = self.r[:n]

        self.kval = int(self.kfrac * n)
        if self.kval != 0:
            ## Pick two disjoint sets of kval species to swap
            kval = np.random.choice(np.arange(n), size=2*self.kval, replace=False)
            kval_in, kval_out = kval[:self.kval], kval[-self.kval:]
            self.kval_in, self.kval_out = kval_in, kval_out

            self.A[kval_in, :] = self.A[kval_out, :] + self.eps * np.random.normal(size=self.A[kval_out, :].shape)
            self.A[:, kval_in] = self.A[:, kval_out] + self.eps * np.random.normal(size=self.A[:, kval_out].shape)
            self.r[kval_in] = self.r[kval_out]

            # P = np.eye(self.A.shape[0])
            # P[kval_in] = P[kval_out] + self.eps * np.random.normal(size=P[kval_out, :].shape)
            # self.P = P
            # self.A = P @ self.A @ P.T
            # self.r = P @ self.r


        # # Find exact coexistence fixed point
        # self.xstar = np.linalg.solve(self.A, -self.r)
        # self.xstar[self.xstar < 0] = 0.0
        # # check stability
        # jac = self.jac(0, self.xstar)
        # eigs = np.linalg.eigvals(jac)

        # # eigs = np.linalg.eigvals(self.A)
        # print(f"Stability observed: {np.all(np.real(eigs) < 0)}")
            
        if self.verbose:
            print(f"Rank observed: {np.linalg.matrix_rank(self.A)}")

    def __call__(self, t, x):
        return x * (self.r + np.matmul(self.A, x))
    
    def jac(self, t, x):
        return np.diag(self.r + np.matmul(self.A, x)) + self.A * x[:, None]
    
    @property
    def integrator_args(self, **kwargs):
        """Default integrator arguments."""

        # Stop based on change
        def stopping_event(t, y, thresh=self.tolerance):
            """Stop when the time derivative is small."""
            dydt = self(t, y)
            return np.linalg.norm(dydt) / np.linalg.norm(y) - thresh
        stopping_event.terminal = self.early_stopping
        self.stopping_event = stopping_event

        ## Calculate a fixed sparsity pattern for the Jacobian
        jac_sparsity = (self.jac(0, np.random.normal(size=self.n)) != 0).astype(int) 
        integrator_args = {
            "events": stopping_event, 
            "jac_sparsity": jac_sparsity,
            "method": "Radau", 
            "atol": self.tolerance, 
            "rtol": self.tolerance, 
            "first_step": self.tolerance
        }
        ## update with user-provided arguments, if given
        integrator_args.update(kwargs)
        return integrator_args