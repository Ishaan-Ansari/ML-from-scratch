import logging 
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClassificationStrategy(ABC):
    @abstractmethod
    def classify(self, df: pd.DataFrame)-> bool:
        """
        Abstract method to apply classification on a DataFrame

        Args:
            df (pd.DataFrame): DataFrame to apply classification

        Returns:
            bool: True if the classification was successful
        """
        pass

# Implementing concrete class for Naive Bayes
class NaiveBayes:
    def __init__(self):
        self._priors = None
        self._likelihoods = None
        self._marginals = None
        self._classes = None
        self._features = None

    def loss(self, y, y_pred):
        y = np.array(y)
        y_pred = np.array(y_pred)
        return np.mean(y != y_pred)

    def fit(self, X, y):
        # TODO: Implement fit
        raise NotImplementedError
    
    def classify(self, X):
        # TODO: Implement classify
        raise NotImplementedError