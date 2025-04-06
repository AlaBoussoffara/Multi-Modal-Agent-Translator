'''
Document Translator Streamlit App
'''

import streamlit as st
import os
import tempfile
import time
from mmat_langgraph_version import langgraph_pipeline  # Updated pipeline import

# --- UI Layout ---
st.set_page_config(page_title="Document Translator", page_icon="🌍")
st.title(":rainbow[Document Translator] 📄🌍")
st.write("Upload a document, select the target language, and get a translated version while preserving the layout.")

# --- Session state defaults ---
if "translation_in_progress" not in st.session_state:
    st.session_state.translation_in_progress = False
if "translation_started" not in st.session_state:
    st.session_state.translation_started = False
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []
if "translated_files" not in st.session_state:
    st.session_state.translated_files = []  # List of dicts: { "name": ..., "path": ... }

# --- Utility Functions ---
def update_log(message: str):
    """
    Append a log message with a timestamp to the session state and update the log display.

    Args:
        message (str): The message to add to the log.
    """
    timestamp = time.strftime("%H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    st.session_state.log_messages.append(full_message)
    print(full_message)
    log_placeholder.text_area("Event Log", "\n".join(st.session_state.log_messages), height=200)

def progress_callback(percent):
    """
    Callback function to update the progress indicators during translation.

    Args:
        percent (int): Current progress percentage.
    """
    progress_bar.progress(percent)
    status_text.text(f"Translation progress: {percent}%")

def start_translation_callback():
    """
    Callback function to mark the translation as started.
    """
    st.session_state.translation_in_progress = True

# --- Language Selection ---
target_language = st.selectbox(
    "Select a language for translation:",
    ["English", "French"],
    index=0
)

# --- File Upload ---
uploaded_files = st.file_uploader("📤 Upload your documents", type=["pdf", "docx"], accept_multiple_files=True)
if uploaded_files:
    st.subheader("Uploaded Files Preview")
    file_info = [
        {"Name": f.name, "Type": f.type, "Size (KB)": round(len(f.getvalue()) / 1024, 2)}
        for f in uploaded_files
    ]
    st.table(file_info)

# --- Start Button ---
start_button = st.button(
    "Start Translation",
    disabled=(not uploaded_files or st.session_state.translation_in_progress),
    on_click=start_translation_callback
)

# --- Progress Container ---
progress_container = st.container()
with progress_container:
    progress_bar = st.progress(0)
    status_text = st.empty()

# --- Create log placeholder early so update_log can always use it ---
log_placeholder = st.empty()

# --- Translation Process ---
if st.session_state.translation_in_progress and not st.session_state.translation_started:
    st.session_state.translation_started = True
    update_log("Starting translation process for uploaded files.")
    
    for uploaded_file in uploaded_files:
        file_name, file_ext = os.path.splitext(uploaded_file.name)
        translated_file_name = f"{file_name}_translated{file_ext}"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_input:
            temp_input.write(uploaded_file.getvalue())
            temp_input_path = temp_input.name
        
        temp_output_path = temp_input_path.replace(file_ext, f"_translated{file_ext}")
        ref_file_path = os.path.join("mt_outputs", uploaded_file.name)
        update_log(f"Processing file: {uploaded_file.name}")
        status_text.text(f"Translating {uploaded_file.name}...")
        
        try:
            evaluation_results = langgraph_pipeline(temp_input_path, temp_output_path, ref_file_path, target_language, progress_callback, False)
            update_log(f"Translation complete for {uploaded_file.name}.")
            # update_log(f"Evaluation results: {evaluation_results[0]['COMET Score']}")
            st.session_state.translated_files.append({
                "name": translated_file_name,
                "path": temp_output_path
            })
        except Exception as e:
            update_log(f"Error translating {uploaded_file.name}: {str(e)}")
        
        os.remove(temp_input_path)
    
    st.session_state.translation_in_progress = False

if st.session_state.translated_files:
    st.success("Translation completed for some files!")
    for file_info in st.session_state.translated_files:
        with open(file_info["path"], "rb") as f:
            st.download_button(
                label=f"Download {file_info['name']}",
                data=f,
                file_name=file_info["name"]
            )
