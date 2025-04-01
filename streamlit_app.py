import streamlit as st
import os
import tempfile
from mmat_langgraph_version import langgraph_pipeline  # Importer le modèle mis à jour

# Configuration de l'application Streamlit
st.set_page_config(page_title="Traducteur de Documents", page_icon="🌍")
st.title(":rainbow[Traducteur de Documents] 📄🌍")
st.write("Téléchargez un document, sélectionnez la langue cible et obtenez une version traduite tout en conservant la mise en page.")

# Sélection de la langue cible
target_language = st.selectbox(
    "Sélectionnez une langue pour la traduction :", 
    ["French", "English"], 
    index=0,  # Sélection par défaut
)

# Téléversement du fichier
uploaded_file = st.file_uploader("📤 Téléchargez votre document", type=["pdf", "docx"])

# Variables de session pour éviter la retraduction
if "translation_done" not in st.session_state:
    st.session_state.translation_done = False
if "translated_file_path" not in st.session_state:
    st.session_state.translated_file_path = None
if "translated_file_name" not in st.session_state:
    st.session_state.translated_file_name = None

# Vérifier si un fichier est téléchargé et que la traduction n'a pas encore été faite
if uploaded_file and not st.session_state.translation_done:
    file_name, file_ext = os.path.splitext(uploaded_file.name)
    translated_file_name = f"{file_name}_translated{file_ext}"

    # Créer un fichier temporaire pour stocker le fichier uploadé
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name

    # Chemin de sortie pour le fichier traduit
    temp_output_path = temp_input_path.replace(file_ext, f"_translated{file_ext}")

    # Affichage de la barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(percent):
        """ Met à jour la barre de progression dynamiquement. """
        progress_bar.progress(percent)
        status_text.text(f"Progression de la traduction : {percent}%")

    with st.status("Traduction en cours... Veuillez patienter ⏳", expanded=False) as status:
        # Exécution de la traduction
        langgraph_pipeline(temp_input_path, temp_output_path, target_language, update_progress)
        update_progress(100)  # Fin de la barre de progression

        # Suppression du fichier temporaire d'entrée après la traduction
        os.remove(temp_input_path)

        status.update(label="Traduction terminée ! 🎉", state="complete", expanded=False)

    # Stocker le chemin du fichier traduit et marquer la traduction comme terminée
    st.session_state.translation_done = True
    st.session_state.translated_file_path = temp_output_path
    st.session_state.translated_file_name = translated_file_name  # Store the translated file name

    # Animation de célébration
    st.balloons()

# Si la traduction est terminée, afficher le bouton de téléchargement
if st.session_state.translation_done and st.session_state.translated_file_path:
    st.success("Votre document traduit est prêt ! 📂")
    with open(st.session_state.translated_file_path, "rb") as f:
        st.download_button("📥 Télécharger le document traduit", f, file_name=st.session_state.translated_file_name)

    # Ne pas supprimer le fichier de sortie après téléchargement pour ne pas relancer la traduction
