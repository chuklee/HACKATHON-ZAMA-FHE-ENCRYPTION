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


class OneVsRest():
    def __init__(self, X, y) -> None:
        self.classes = np.unique(y)
        self.models = {}
        self.X = X
        self.y = y

    def getModels(self):
        return self.models

    def fit(self):
        parameters_range = (-1.0, 1.0)
        for cl in self.classes:
            binary_y = np.where(self.y == cl, 0, 1)
            print(binary_y)
            X_train, X_test, y_train, y_test = train_test_split(
                self.X,
                binary_y,
                test_size=0.2,
                random_state=42,
                stratify=binary_y
            )

            fhe_model = SGDClassifier(
                random_state=42,
                max_iter=15,
                fit_encrypted=True,
                parameters_range=parameters_range,
                verbose=True,
            )

            print(y_train)

            fhe_model.fit(X_train, y_train, fhe="execute")
            fhe_model.compile(X_train)
            self.models[cl] = fhe_model
            y_pred_fhe = fhe_model.predict(X_test, fhe="execute")
            accuracy_fhe = accuracy_score(y_test, y_pred_fhe)
            print(f"Accuracy for class {cl} vs the rest: {accuracy_fhe:.2f}")


if __name__ == "__main__":
    X = np.load("data/facenet_olivetti/x.npy")
    y = np.load("data/facenet_olivetti/y.npy")


    # 5. Entraînement et évaluation de la Régression Logistique
    X_facenet_rms =  np.sqrt(np.mean(X**2, axis=1))
    X_facenet_mean = np.mean(X, axis=1)
    X_facenet_median = np.median(X, axis=1)

    X_facenet_features = np.stack([X_facenet_rms, X_facenet_mean, X_facenet_median], axis=1)
    one_vs_rest = OneVsRest(X_facenet_features, y)
    one_vs_rest.fit()
