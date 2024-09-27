from typing import Optional
from concrete.compiler import parameter
from pathlib import Path
from tempfile import TemporaryDirectory
from concrete.ml import deployment
import numpy as np
from concrete import fhe
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import SGDClassifier
from preprocess import load_dataset, featurisation
from sklearn.model_selection import train_test_split
import os


class ModelTrainer():
    def __init__(self, target_folder_path: str, test_size=0.2) -> None:
        self.target_folder_path = target_folder_path
        self.fhe_model = np.load(f"{target_folder_path}/fhe_model.npy") if os.path.exists(
            f"{target_folder_path}/fhe_model.npy") else None
        self.embeddings, self.labels = load_dataset(target_folder_path)
        self.features = featurisation(self.embeddings)
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(
            self.features, self.labels, test_size)

    def split_data(self, X: np.ndarray, y: np.ndarray, test_size) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test

    def fit(self, sgd_classifier: SGDClassifier) -> None:
        sgd_classifier.fit(self.X_train, self.y_train, fhe="execute")
        sgd_classifier.compile(self.X_train)
        self.fhe_model = sgd_classifier

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

    def save(self, path: Optional[str] = None) -> None:
        if path is None:
            path = self.target_folder_path
        np.save(f"{path}/fhe_model.npy", self.fhe_model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.fhe_model.predict(X)

    def accuray(self, X: Optional[np.ndarray]=None, y: Optional[np.ndarray]=None) -> float:
        if X is None or y is None:
            X = self.X_test
            y = self.y_test
        if self.fhe_model is None:
            raise ValueError("Model not trained")
        return self.fhe_model.score(X, y)


if __name__ == "__main__":
    # Instantiate cluster.
    model_trainer = ModelTrainer("./data/lfw_people/George_HW_Bush", None)
    if model_trainer.fhe_model is None:
        sgd_classifier = SGDClassifier(
            random_state=42,
            max_iter=15,
            fit_encrypted=True,
            parameters_range=(-1.0, 1.0),
            verbose=True,
        )
        model_trainer.fit(sgd_classifier)
    # model_trainer.save()
    print(model_trainer.accuray())
