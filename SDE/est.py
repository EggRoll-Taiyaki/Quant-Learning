from typing import Callable, Optional, Any
import numpy as np
from scipy.optimize import minimize

from sde import *
from adapter import *

class SDEEstimator:
    
    """
        General-purpose estimator for stochastic differential equation (SDE) models
        using Maximum Likelihood Estimation (MLE) or Maximum A Posteriori Estimation (MAP).

        ---------------------------------------
        Mathematical Formulation:
    
        Given observed data D and model parameters θ, this estimator minimizes the 
        negative log-posterior:

            θ̂ = argmax_θ  log p(D | θ)         ← MLE
            θ̂ = argmax_θ  [log p(D | θ) + log p(θ)]  ← MAP
        ---------------------------------------

        Args:
            model_class (Callable[..., Any]):
                A class or function that accepts a flat parameter vector (via *args)
                and returns a model instance that implements .log_likelihood(data).

            adapter (Callable[..., Any]):
                To pack and unpack params for the targeted model class

            prior (Optional[Callable[[np.ndarray], float]]):
                A log-prior function. If provided, it should return log p(θ) given θ.
                Else, the estimator performs MLE.
    """

    def __init__(
        self,
        model_class: Callable[..., Any],
        adapter    : Callable[..., Any],
        prior      : Optional[Callable[[np.ndarray], float]] = None
    ):

        self.model_class = model_class
        self.adapter     = adapter(model_class)
        self.prior       = prior

    def _negative_log_posterior(
        self, 
        params: np.ndarray, 
        data  : Any
    ) -> float:
    
        """
            Compute the negative log-posterior given parameters and data.

            Args:
                params (np.ndarray): Parameter vector θ.
                data   (Any)       : Observed data used in the likelihood.

            Returns:
                p (float): Negative log-posterior/likelihood
        """

        model = self.model_class(*self.adapter.unpack(params))
        log_lik   = model.log_likelihood(data)
        log_prior = self.prior(params) if self.prior else 0.0
        return -(log_lik + log_prior)

    def fit(
        self,
        data      : Any,
        init_guess: np.ndarray,
        bounds    : Optional[list[tuple[float, float]]] = None,
        method    : str = "L-BFGS-B"
    ) -> np.ndarray:

        """
            Fit model parameters via numerical optimization of the negative log-posterior.

            Args:
                data       (Any)                  : Observation data (e.g., list of (v, t) tuples).
                init_guess (np.ndarray)           : Initial parameter vector.
                bounds     (Optional[list[tuple]]): Parameter bounds for optimizer.
                method     (str)                  : Optimization algorithm (default: "L-BFGS-B").

            Returns:
                best_params (np.ndarray): Estimated parameter vector.
        """
        
        result = minimize(
            fun    = lambda p: self._negative_log_posterior(p, data),
            x0     = init_guess,
            method = method,
            bounds = bounds,
        )

        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)

        return result.x

if __name__ == '__main__':

    mu = np.array([3.5, 4.2])
    sigma = np.array([
        [0.2, 0.3],
        [0.3, 0.4]
    ])

    gbm = GeometricBrownianMotion(mu = mu, sigma = sigma)
    path = gbm.simulate(x0 = np.array([100.0, 100.0]), T = 1.0, n = 1000)

    estimator = SDEEstimator(
        GeometricBrownianMotion,
        GBMParamAdapter
    )

    best_params = estimator.fit(
        path,
        np.array([0, 0, 1.0, 0, 1.0])
    )

    gbm_adapter = GBMParamAdapter(GeometricBrownianMotion)
    fake_gbm = GeometricBrownianMotion(*gbm_adapter.unpack(best_params))

    print(f"μ  = {gbm.mu}")
    print(f"μ' = {fake_gbm.mu}")
    print(f"σ = {gbm.sigma}")
    print(f"σ' = {fake_gbm.sigma}")

    fake_path = fake_gbm.simulate(x0 = np.array([100.0, 100.0]), T = 1.0, n = 1000)

    import matplotlib.pyplot as plt

    def compare_multidim_paths(path1, path2):

        states      = np.array([x for x, _ in path])
        fake_states = np.array([x for x, _ in fake_path])
        times       = [t for _, t in path]

        assert states.shape[1] == fake_states.shape[1], "Two paths should have same dimension."

        for i in range(states.shape[1]):
            plt.figure(figsize=(10, 5))
            plt.plot(times, states[:, i], label=f"Dimension {i} of Path1")
            plt.plot(times, fake_states[:, i], label=f"Fake Dimension {i} of Path2")

            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.title("Multi-dimensional SDE Path")
            plt.legend()
            plt.show()

    compare_multidim_paths(path, fake_path)
