import os
from concrete.compiler import parameter
import cv2
import tempfile
import numpy as np
import pandas as pd
from torch.utils.data import random_split
from tqdm import tqdm
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from deepface import DeepFace
import xgboost as xgb
import warnings
from concrete import fhe
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import SGDClassifier
from concrete.ml.sklearn.base import SklearnSGDClassifierMixin


class ModelsCluster():
    def __init__(self) -> None:
        self.models = {}

    def getModel(self, pubkey: str) -> dict[str, SklearnSGDClassifierMixin]:
        return self.models[pubkey]

    def fit(self, X, y, pubkey: str) -> None:
        parameters_range = (-1.0, 1.0)

        fhe_model = SGDClassifier(
            random_state=42,
            max_iter=15,
            fit_encrypted=True,
            parameters_range=parameters_range,
            verbose=True,
        )
        fhe_model.fit(X, y, fhe="execute")
        fhe_model.compile(X)
        self.models[pubkey] = fhe_model




if __name__ == "__main__":
    cluster = ModelsCluster()

    X = np.load("data/facenet_olivetti/x.npy")
    y = np.load("data/facenet_olivetti/y.npy")

    X_facenet_rms =  np.sqrt(np.mean(X**2, axis=1))
    X_facenet_mean = np.mean(X, axis=1)
    X_facenet_median = np.median(X, axis=1)

    X_facenet_features = np.stack([X_facenet_rms, X_facenet_mean, X_facenet_median], axis=1)

    cluster.fit(X_facenet_features, y, "HERE IS MY PUBLIC KEY")
