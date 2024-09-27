import numpy as np
from tqdm import tqdm
from glob import glob
import os


def load_image_folder(folder_path: str, max_nb_images: int) -> list[str]:
    image_path = f"{folder_path}/*"
    image_paths = glob(image_path)
    images = [image_path for image_path in image_paths[:max_nb_images]]
    return images


def compute_embeddings_and_labels(images: list[str], label: int) -> np.ndarray:
    from deepface import DeepFace
    embeddings = []
    labels = []
    for image in tqdm(images):
        try:
            embedding_obj = DeepFace.represent(
                img_path=image,
                model_name="Facenet",
            )
            embedding = embedding_obj[0]['embedding']
            embeddings.append(embedding)
            labels.append(label)
        except Exception as e:
            print(f"Erreur lors du traitement de {image}: {e}")
    return np.vstack(embeddings), np.array(labels)


def load_embeddings_and_labels(folder_path: str, label: int, max_nb_images: int, cache: bool) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(f"{folder_path}/embeddings.npy") or not os.path.exists(f"{folder_path}/labels.npy") or (not cache):
        images = load_image_folder(folder_path, max_nb_images)
        embeddings, labels = compute_embeddings_and_labels(images, label)
        np.save(f"{folder_path}/embeddings.npy", embeddings)
        np.save(f"{folder_path}/labels.npy", labels)
    embeddings = np.load(f"{folder_path}/embeddings.npy")
    labels = np.load(f"{folder_path}/labels.npy")
    return embeddings, labels


def load_dataset(target_folder: str, max_nb_images=500, cache=True, deep_fake_folder: str = "./data/deepfake") -> tuple[np.ndarray, np.ndarray]:
    deep_fake_images_embeddings, deep_fake_labels = load_embeddings_and_labels(
        deep_fake_folder, max_nb_images=max_nb_images, label=1, cache=cache)
    target_images_embeddings, target_images_labels = load_embeddings_and_labels(
        target_folder, max_nb_images=max_nb_images, label=0, cache=cache)
    embeddings = np.vstack(
        [target_images_embeddings, deep_fake_images_embeddings])
    labels = np.hstack([target_images_labels, deep_fake_labels])
    return embeddings, labels


if __name__ == "__main__":
    embeddings, labels = load_dataset(
        "./data/lfw_people/George_HW_Bush", cache=True)
