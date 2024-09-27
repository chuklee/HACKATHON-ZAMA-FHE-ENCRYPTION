from concrete.compiler import parameter
from pathlib import Path
from tempfile import TemporaryDirectory
from concrete.ml import deployment
import numpy as np
from concrete import fhe
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import SGDClassifier
from deepface import DeepFace


class ModelTrainer():
    def __init__(self, deploy_path: str) -> None:
        self.model = None
        self.facenet = DeepFace.build_model("Facenet")
        self.deployment_path = deploy_path

    def fit(self, X: np.ndarray, y: np.ndarray, id: str) -> None:
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

        self.model = fhe_model

    # def deployment(self):
    #     DEPLOYMENT_PATH = Path(self.deployment_path)
    #     DEPLOYMENT_PATH.mkdir(exist_ok=True)
    #     deployment_dir = TemporaryDirectory(dir=str(DEPLOYMENT_PATH))
    #     deployment_path = Path(deployment_dir.name)
    #
    #     fhe_dev = FHEModelDev(deployment_path, self.model)
    #     fhe_dev.save(mode="training")
    #
    #     fhe_client = FHEModelClient(deployment_path)
    #     fhe_client.load()
    #     serialized_evaluation_keys = fhe_client.get_serialized_evaluation_keys()


if __name__ == "__main__":
    # Registration
    """
    Here we receive the pictures of the user wanting to register.
    We crop then embedd the pictures and merge them into the existing dataset.
    Once this is done we train the model and bind it to it's public key.
    """

    # Instantiate cluster.
    cluster = ModelTrainer("fhe_training")

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
    id = ""
    cluster.fit(X_facenet_features, y, id)

    ################### TRAINING #####################

    # Run inference on encrypted data, Listen on server - TODO
    image = np.array([1, 2, 3])
    # response = cluster.predict(id, image)

    # Send response back to user - TODO

    ################### INFERENCE #####################
