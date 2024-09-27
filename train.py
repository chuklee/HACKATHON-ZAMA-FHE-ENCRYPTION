from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
import tempfile
from deepface import DeepFace


olivetti = fetch_olivetti_faces()
images = olivetti.images  # Forme : (400, 64, 64)
labels = olivetti.target  # Forme : (400,)


temp_dir = tempfile.mkdtemp()
image_paths = []

for idx, img in enumerate(images):
    img_uint8 = (img * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    img_path = os.path.join(temp_dir, f"img_{idx}.png")
    cv2.imwrite(img_path, img_rgb)
    image_paths.append(img_path)
    
    

model_name = "Facenet512"  

embeddings = []

for img_path in tqdm(image_paths, desc="Extraction des embeddings"):
    try:
        embedding_obj = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            enforce_detection=False 
        )
        embedding = embedding_obj[0]['embedding']
        embeddings.append(embedding)
    except Exception as e:
        print(f"Erreur lors du traitement de {img_path}: {e}")
        embeddings.append(np.zeros(512))  


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from concrete import fhe
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import SGDClassifier

X = pd.DataFrame(embeddings)
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y)),
    eval_metric='mlogloss',
    use_label_encoder=False
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
