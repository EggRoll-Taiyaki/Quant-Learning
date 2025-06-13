from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class ParamAdapterBase(ABC):

    def __init__(
        self, 
        model_class: type
    ):
        
        self.model_class = model_class

    @abstractmethod
    def pack(
        self, 
        *args
    ) -> np.ndarray:
        
        """
            Convert model parameters to flat vector.
        """
        raise NotImplementedError("Packing not implemented.")

    @abstractmethod
    def unpack(
        self, 
        params: np.ndarray
    ) -> tuple:
        
        """
            Convert flat parameter vector into structured args for model.
        """
        raise NotImplementedError("Unpacking not implemented.")

class GBMParamAdapter(ParamAdapterBase):
    
    def pack(
        self, 
        mu   : np.ndarray, 
        sigma: np.ndarray
    ) -> np.ndarray:
        
        ndim = mu.shape[0]

        flat_sigma = sigma[np.tril_indices(ndim)]
        return np.concatenate([mu, flat_sigma])

    def unpack(
        self, 
        params: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
 
        ndim = int((-3 + np.sqrt(9 + 8 * params.shape[0])) / 2)   

        mu         = params[:ndim]
        flat_sigma = params[ndim:]

        sigma      = np.zeros((ndim, ndim))
        sigma[np.tril_indices(ndim)] = flat_sigma
        sigma = sigma + sigma.T - np.diag(np.diag(sigma))
        return mu, sigma


