import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from plyer import notification
import json

# 📦 Chargement du modèle
model = tf.keras.models.load_model("cnn_reseau_5x5.h5")

# 📂 Chargement des noms de classes
with open("classes.json", "r") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

# ⚙️ Paramètres
TAILLE_IMAGE = (5, 5)

# 🛡️ Titre de l'application
st.title("🛡️ Détection d'intrusion réseau par CNN")
st.write("Chargez une image représentant un flux réseau pour identifier une éventuelle attaque.")

# 📤 Upload de l'image
uploaded_file = st.file_uploader("📤 Charger une image PNG (5x5)", type=["png"])

if uploaded_file is not None:
    # 🖼️ Affichage de l'image
    image = Image.open(uploaded_file).convert("L").resize(TAILLE_IMAGE)
    st.image(image, caption="Image du flux réseau", use_column_width=False)

    # 🔄 Prétraitement
    img_array = np.array(image).reshape(1, TAILLE_IMAGE[0], TAILLE_IMAGE[1], 1) / 255.0

    # 🔍 Prédiction
    prediction = model.predict(img_array)
    classe_predite = np.argmax(prediction)
    nom_classe = index_to_class[classe_predite]
    score = prediction[0][classe_predite]

    # 📊 Affichage du résultat
    st.markdown(f"### 🧠 Classe prédite : `{nom_classe}`")
    st.markdown(f"Confiance du modèle : **{score:.2%}**")

    # 🚨 Notification push si attaque détectée
    if "BENIGN" not in nom_classe.upper():
        notification.notify(
            title="🚨 Intrusion détectée",
            message=f"Type : {nom_classe} | Confiance : {score:.2%}",
            timeout=5
        )
        st.warning("⚠️ Alerte : Intrusion détectée ! Une notification a été envoyée.")
    else:
        st.success("✅ Aucun comportement malveillant détecté.")