import streamlit as st
import os
import tempfile
from mmat_langgraph_version import langgraph_pipeline  # Importer le modèle mis à jour

# Configuration de l'application Streamlit
st.set_page_config(page_title="Traducteur de Documents", page_icon="🌍")
st.title(":rainbow[Traducteur de Documents] 📄🌍")
st.write("Téléchargez un document, sélectionnez la langue cible et obtenez une version traduite tout en conservant la mise en page.")

# Sélection de la langue cible
st.subheader("🌍 Choisissez la langue cible")
target_language = st.selectbox(
    "Sélectionnez une langue pour la traduction :", 
    ["French", "English"], 
    index=0,  # Sélection par défaut
)

# Téléversement du fichier
uploaded_file = st.file_uploader("📤 Téléchargez votre document", type=["pdf", "docx"])

# Vérifier si un fichier a été téléchargé
if uploaded_file:
    file_name, file_ext = os.path.splitext(uploaded_file.name)
    translated_file_name = f"{file_name}_translated{file_ext}"

    # Bouton pour lancer la traduction
    if st.button("🚀 Lancer la traduction"):
        # Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(percent):
            """ Met à jour la barre de progression Streamlit dynamiquement selon la sortie de LangGraph """
            progress_bar.progress(percent)
            status_text.text(f"Progression de la traduction : {percent}%")

        with st.status("Traduction en cours... Veuillez patienter ! ⏳", expanded=False) as status:
            # Création d'un fichier temporaire pour l'entrée
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_input:
                temp_input.write(uploaded_file.read())
                temp_input_path = temp_input.name

            # Création du chemin de sortie temporaire
            temp_output_path = temp_input_path.replace(file_ext, f"_translated{file_ext}")

            # Exécution de la traduction avec la langue sélectionnée
            langgraph_pipeline(temp_input_path, temp_output_path, target_language, update_progress)
            # Mise à jour de la barre de progression à 100%
            update_progress(100)
            # Suppression du fichier d'entrée après le traitement
            os.remove(temp_input_path)

            status.update(label="Traduction terminée ! 🎉", state="complete", expanded=False)

        # Bouton de téléchargement du document traduit
        st.success("Votre document traduit est prêt ! 📂")
        with open(temp_output_path, "rb") as f:
            st.download_button("📥 Télécharger le document traduit", f, file_name=translated_file_name)

        # Suppression du fichier de sortie après téléchargement
        os.remove(temp_output_path)

        # Animation de célébration
        st.balloons()
