from concrete.compiler import parameter
import numpy as np
from concrete import fhe
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import SGDClassifier
from concrete.ml.sklearn.base import SklearnSGDClassifierMixin


class ModelsCluster():
    def __init__(self) -> None:
        self.models = {}

    def getModel(self, pubkey: str) -> SklearnSGDClassifierMixin:
        return self.models[pubkey]

    def fit(self, X: np.ndarray, y: np.ndarray, pubkey: str) -> None:
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

    def predict(self, pubkey: str, emb_image: np.ndarray) -> bool:
        model = self.getModel(pubkey)
        res = model.predict(emb_image, fhe="execute")
        return res[0] == 0


if __name__ == "__main__":
    # Registration
    """
    Here we receive the pictures of the user wanting to register.
    We crop then embedd the pictures and merge them into the existing dataset.
    Once this is done we train the model and bind it to it's public key.
    """

    # Instantiate cluster.
    cluster = ModelsCluster()

    ################### DATASET #####################

    X = np.load("data/facenet_olivetti/x.npy")
    y = np.load("data/facenet_olivetti/y.npy")

    X_facenet_rms =  np.sqrt(np.mean(X**2, axis=1))
    X_facenet_mean = np.mean(X, axis=1)
    X_facenet_median = np.median(X, axis=1)

    X_facenet_features = np.stack([X_facenet_rms, X_facenet_mean, X_facenet_median], axis=1)

    ################### DATASET #####################

    ################### TRAINING #####################
    # Train a model for new client.
    pubkey = ""
    cluster.fit(X_facenet_features, y, pubkey)

    ################### TRAINING #####################

    # Run inference on encrypted data, Listen on server - TODO
    image = np.array([1, 2, 3])
    response = cluster.predict(pubkey, image)

    # Send response back to user - TODO

    ################### INFERENCE #####################
