from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
   
class StochasticDifferentialEquation(ABC):
    
    """
        Abstract base class for Stochastic Differential Equations (SDEs).
            
            dx = μ(t, x) dt + σ(t, x) dW(t)

        Custom SDEs must implement `drift` and `diffusion` terms.
    """
    
    @abstractmethod
    def drift(
        self, 
        t: float, 
        x: np.ndarray
    ) -> np.ndarray:

        """
            Compute the drift term μ(t, x) of the SDE.
        
            Args:
                t (float)     : Time
                x (np.ndarray): Current state vector (e.g., [stock_price, volatility] for Heston)
            
            Returns:
                v (np.ndarray): Drift vector
        """
        
        pass
    
    @abstractmethod
    def diffusion(
        self, 
        t: float, 
        x: np.ndarray
    ) -> np.ndarray:

        """
            Compute the diffusion term σ(t, x) of the SDE.
        
            Args
                t (float)     : Time
                x (np.ndarray): Current state vector

            Returns:
                m (np.ndarray): Diffusion matrix (for multi-dimensional SDEs)
        """
        
        pass
 
    def log_likelihood(
        self,
        data: list[tuple[float, float]]
    ) -> Optional[float]:

        """
            Compute the log-likelihood of observing `data` 
            Override this if applicable

            Args:
                data (list of (float, float)): pairs of (value, time) 

            Returns:
                p (float): posterior probability 

        """

        raise NotImplementedError("This model does not support log-likelihood computation.")

    def simulate(
        self,
        x0: np.ndarray,
        T : float,
        n : int = 252,
    ) -> list[tuple[np.ndarray, float]]:
        
        """
            Simulation of the SDE using Euler-Maruyama discretization.
        
            Args:
                x0 (np.ndarray): Initial state vector
                T (float)      : Total time horizon
                n (int)        : Number of time steps
            
            Returns:
                list of tuples 
                    - np.ndarray : sample stock prices 
                    - float      : sample time
        """
        
        dt  = T / n
        dim = len(x0)

        path = [(x0, 0)]
        
        for t in range(1, n + 1):

            cur = path[t - 1][0]
            dWt = np.random.normal(0, np.sqrt(dt), size=dim)
            
            path += [(
                cur
                    + self.drift(t * dt, cur) * dt 
                    + self.diffusion(t * dt, cur) @ dWt
            , t * dt)]
                
        return path

class GeometricBrownianMotion(StochasticDifferentialEquation):

    """
        Geometric Brownian Motion defined by

            dS_t = μ S_t dt + σ S_t dW_t

        μ, σ are supposed to be constant
    """
    
    def __init__(
        self,
        mu   : np.ndarray, 
        sigma: np.ndarray
    ):

        assert mu.ndim    == 1
        assert sigma.ndim == 2
        assert mu.shape[0] == sigma.shape[0] == sigma.shape[1]

        self.mu    = mu
        self.sigma = sigma

    def drift(
        self,
        t: float, 
        x: np.ndarray
    ) -> np.ndarray:
        
        return self.mu * x # element-wise multiplication
    
    def diffusion(
        self, 
        t: float, 
        x: np.ndarray
    ) -> np.ndarray:
        
        return self.sigma @ np.diag(x)

    def log_likelihood(
        self,
        data: list[tuple[np.ndarray, float]]
    ) -> float:

        # --- Extract and validate input ---
        
        prices, times = zip(*data)

        prices = np.stack(prices)
        times  = np.array(times) 

        log_returns = np.log(prices[1:] / prices[:-1])
        dts         = times[1:] - times[:-1]

        d = len(self.mu)
        assert prices.shape[1] == d, "Dimension mismatch between prices and model"

        # --- Precompute covariance matrix, log-det and inverse ---
        
        mu, sigma  = self.mu, self.sigma
        cov        = sigma @ sigma.T
        _, log_det = np.linalg.slogdet(cov)
        inv_cov    = np.linalg.inv(cov)

        # --- Precompute constant term ---
        constant = d * np.log(2 * np.pi)

        log_prob = 0.0

        for log_return, dt in zip(log_returns, dts):

            mean = (mu - 0.5 * np.diag(cov)) * dt
            
            # --- Multivariate normal log-likelihood ---
            #   log p(x) = -0.5 * (log(det(2πΣ)) + (x - μ)^T Σ^{-1} (x - μ))

            diff = log_return - mean
            quad = diff @ (1.0 / dt * inv_cov) @ diff

            log_prob += -0.5 * (constant + log_det + quad)

        return log_prob

if __name__ == '__main__':

    mu = np.array([0.05, 0.03])
    sigma = np.array([
        [0.2, 0.1],
        [0.1, 0.3]
    ])

    gbm = GeometricBrownianMotion(mu = mu, sigma = sigma)
    path = gbm.simulate(x0 = np.array([100.0, 100.0]), T = 1.0, n = 1000)

    import matplotlib.pyplot as plt

    def plot_multidim_path(path: list[tuple[np.ndarray, float]]):

        times  = [t for _, t in path]
        states = np.array([x for x, _ in path])
        dim = states.shape[1]

        plt.figure(figsize=(10, 5))
        for i in range(dim):
            plt.plot(times, states[:, i], label=f"Dimension {i}")

        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Multi-dimensional SDE Path")
        plt.legend()
        plt.show()

    plot_multidim_path(path)
