'''
Application Streamlit pour la traduction de documents
'''

import streamlit as st
import os
import tempfile
import time
from mmat_langgraph_version import langgraph_pipeline  # Import du pipeline mis à jour

# --- Mise en page de l'interface ---
st.set_page_config(page_title="Traducteur de Documents", page_icon="🌍")
st.title(":rainbow[Traducteur de Documents] 📄🌍")
st.write("Téléchargez un document, sélectionnez la langue cible, et obtenez une version traduite tout en préservant la mise en page.")

# --- États par défaut de la session ---
if "translation_in_progress" not in st.session_state:
    st.session_state.translation_in_progress = False
if "translation_started" not in st.session_state:
    st.session_state.translation_started = False
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []
if "translated_files" not in st.session_state:
    st.session_state.translated_files = []  # Liste de dictionnaires : { "name": ..., "path": ... }

# --- Fonctions utilitaires ---
def update_log(message: str):
    """
    Ajoute un message de journal avec un horodatage à l'état de session et met à jour l'affichage du journal.

    Args:
        message (str): Le message à ajouter au journal.
    """
    timestamp = time.strftime("%H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    st.session_state.log_messages.append(full_message)
    print(full_message)
    log_placeholder.text_area("Journal des événements", "\n".join(st.session_state.log_messages), height=200)

def progress_callback(percent):
    """
    Fonction de rappel pour mettre à jour les indicateurs de progression pendant la traduction.

    Args:
        percent (int): Pourcentage de progression actuel.
    """
    progress_bar.progress(percent)
    status_text.text(f"Progression de la traduction : {percent}%")

def start_translation_callback():
    """
    Fonction de rappel pour marquer le début de la traduction.
    """
    st.session_state.translation_in_progress = True

# --- Sélection de la langue ---
target_language = st.selectbox(
    "Sélectionnez une langue pour la traduction :",
    ["Anglais", "Français"],
    index=0
)

# --- Téléchargement de fichiers ---
uploaded_files = st.file_uploader("📤 Téléchargez vos documents", type=["pdf", "docx"], accept_multiple_files=True)
if uploaded_files:
    st.subheader("Aperçu des fichiers téléchargés")
    file_info = [
        {"Nom": f.name, "Type": f.type, "Taille (Ko)": round(len(f.getvalue()) / 1024, 2)}
        for f in uploaded_files
    ]
    st.table(file_info)

# --- Bouton de démarrage ---
start_button = st.button(
    "Démarrer la traduction",
    disabled=(not uploaded_files or st.session_state.translation_in_progress),
    on_click=start_translation_callback
)

# --- Conteneur de progression ---
progress_container = st.container()
with progress_container:
    progress_bar = st.progress(0)
    status_text = st.empty()

# --- Créez un espace réservé pour le journal ---
log_placeholder = st.empty()

# --- Processus de traduction ---
if st.session_state.translation_in_progress and not st.session_state.translation_started:
    st.session_state.translation_started = True
    update_log("Début du processus de traduction pour les fichiers téléchargés.")
    
    for uploaded_file in uploaded_files:
        file_name, file_ext = os.path.splitext(uploaded_file.name)
        translated_file_name = f"{file_name}_traduit{file_ext}"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_input:
            temp_input.write(uploaded_file.getvalue())
            temp_input_path = temp_input.name
        
        temp_output_path = temp_input_path.replace(file_ext, f"_traduit{file_ext}")
        ref_file_path = os.path.join("mt_outputs", uploaded_file.name)
        update_log(f"Traitement du fichier : {uploaded_file.name}")
        status_text.text(f"Traduction de {uploaded_file.name} en cours...")
        
        try:
            evaluation_results = langgraph_pipeline(temp_input_path, temp_output_path, ref_file_path, target_language, progress_callback, False)
            update_log(f"Traduction terminée pour {uploaded_file.name}.")
            # update_log(f"Résultats de l'évaluation : {evaluation_results[0]['COMET Score']}")
            st.session_state.translated_files.append({
                "name": translated_file_name,
                "path": temp_output_path
            })
        except Exception as e:
            update_log(f"Erreur lors de la traduction de {uploaded_file.name} : {str(e)}")
        
        os.remove(temp_input_path)
    
    st.session_state.translation_in_progress = False

if st.session_state.translated_files:
    st.success("Traduction terminée pour certains fichiers !")
    for file_info in st.session_state.translated_files:
        with open(file_info["path"], "rb") as f:
            st.download_button(
                label=f"Télécharger {file_info['name']}",
                data=f,
                file_name=file_info["name"]
            )
