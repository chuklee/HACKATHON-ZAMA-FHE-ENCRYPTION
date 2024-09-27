import os
import cv2
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from deepface import DeepFace
import xgboost as xgb
import warnings

# Supprimer les avertissements de DeepFace concernant les versions
warnings.filterwarnings("ignore")

# Liste des modèles à tester
models = [
    "VGG-Face", 
    "Facenet", 
    "Facenet512", 
    "OpenFace", 
    "DeepFace", 
    "DeepID", 
    "ArcFace", 
    "Dlib",
    "SFace",
    "GhostFaceNet",
]

# 1. Charger le jeu de données Olivetti Faces
print("Chargement du jeu de données Olivetti Faces...")
olivetti = fetch_olivetti_faces()
images = olivetti.images  # Forme : (400, 64, 64)
labels = olivetti.target  # Forme : (400,)

# 2. Prétraitement des images
print("Prétraitement des images...")
temp_dir = tempfile.mkdtemp()
image_paths = []

for idx, img in enumerate(images):
    # Convertir l'image en uint8
    img_uint8 = (img * 255).astype(np.uint8)
    # Convertir en RGB en dupliquant le canal gris
    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    # Sauvegarder l'image temporairement
    img_path = os.path.join(temp_dir, f"img_{idx}.png")
    cv2.imwrite(img_path, img_rgb)
    image_paths.append(img_path)

print(f"Images sauvegardées dans le répertoire temporaire: {temp_dir}")

# Initialiser des listes pour stocker les résultats et les tailles d'embeddings
results = []
embedding_sizes = []

# 3. Itérer sur chaque modèle pour extraire les embeddings et évaluer les classificateurs
for model_name in models:
    print(f"\nTraitement avec le modèle: {model_name}")
    try:
        embeddings = []
        # Extraction des embeddings pour toutes les images avec le modèle actuel
        for img_path in tqdm(image_paths, desc=f"Extraction des embeddings avec {model_name}"):
            embedding_obj = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                enforce_detection=False
            )
            # Vérifier si l'embedding est récupéré correctement
            if embedding_obj and 'embedding' in embedding_obj[0]:
                embedding = embedding_obj[0]['embedding']
                embeddings.append(embedding)
            else:
                # En cas d'échec, ajouter un vecteur nul de taille appropriée
                # Essayer de déterminer la taille de l'embedding via le modèle
                try:
                    model = DeepFace.build_model(model_name)
                    vector_size = model.layers[-1].output_shape[-1]
                    embeddings.append(np.zeros(vector_size))
                except:
                    # Si la taille ne peut pas être déterminée, utiliser une taille par défaut
                    embeddings.append(np.zeros(512))  # Par exemple, 512
    
        embeddings = np.array(embeddings)
        embedding_size = embeddings.shape[1]  # Taille des embeddings
        embedding_sizes.append({'Modèle': model_name, 'Taille de l\'embedding': embedding_size})
        print(f"Extraction des embeddings avec {model_name} réussie. Taille de l'embedding: {embedding_size}")
        
    except ValueError as ve:
        print(f"Le modèle {model_name} n'est pas supporté par DeepFace. Skipping...")
        continue
    except Exception as e:
        print(f"Une erreur est survenue avec le modèle {model_name}: {e}")
        continue

    # 4. Préparation des données pour les classificateurs
    X = embeddings
    y = labels

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Entraînement et évaluation de la Régression Logistique
    try:
        print("Entraînement de la Régression Logistique...")
        lr_model = LogisticRegression(max_iter=1000, multi_class='auto', solver='lbfgs')
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        accuracy_lr = accuracy_score(y_test, y_pred_lr) * 100
        print(f"Précision de la Régression Logistique: {accuracy_lr:.2f}%")
        results.append({
            'Modèle': model_name,
            'Classificateur': 'Régression Logistique',
            'Précision (%)': round(accuracy_lr, 2)
        })
    except Exception as e:
        print(f"Erreur lors de l'entraînement de la Régression Logistique avec le modèle {model_name}: {e}")
        results.append({
            'Modèle': model_name,
            'Classificateur': 'Régression Logistique',
            'Précision (%)': 'Erreur'
        })

    # 6. Entraînement et évaluation de XGBoost
    try:
        print("Entraînement de XGBoost...")
        xgb_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(np.unique(y)),
            eval_metric='mlogloss',
            use_label_encoder=False,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb) * 100
        print(f"Précision de XGBoost: {accuracy_xgb:.2f}%")
        results.append({
            'Modèle': model_name,
            'Classificateur': 'XGBoost',
            'Précision (%)': round(accuracy_xgb, 2)
        })
    except Exception as e:
        print(f"Erreur lors de l'entraînement de XGBoost avec le modèle {model_name}: {e}")
        results.append({
            'Modèle': model_name,
            'Classificateur': 'XGBoost',
            'Précision (%)': 'Erreur'
        })

# 4. Compilation et Affichage des Résultats

# Créer DataFrame pour les résultats de classification
results_df = pd.DataFrame(results)

# Créer DataFrame pour les tailles d'embeddings
embedding_sizes_df = pd.DataFrame(embedding_sizes)

print("\nRésultats des classificateurs:")
print(results_df.to_string(index=False))

print("\nTailles des embeddings par modèle:")
print(embedding_sizes_df.to_string(index=False))