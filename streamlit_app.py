import streamlit as st
import os
import tempfile
from mmat_langgraph_version import langgraph_pipeline  # Import updated model

# Streamlit App Configuration
st.set_page_config(page_title="Document Translator", page_icon="🌍")
st.title(":rainbow[Document Translator] 📄🌍")
st.write("Upload a document, and get a translated version while keeping the layout intact.")

uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx"])

if uploaded_file:
    file_name = uploaded_file.name
    file_ext = os.path.splitext(file_name)[-1]

    # Progress bar UI
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(percent):
        """ Update Streamlit progress bar dynamically based on LangGraph's output """
        progress_bar.progress(percent)
        status_text.text(f"Translation Progress: {percent}%")

    with st.status("Translating... Please wait! ⏳", expanded=False) as status:
        # Create temporary input file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_input:
            temp_input.write(uploaded_file.read())
            temp_input_path = temp_input.name

        # Create temporary output file path
        temp_output_path = temp_input_path.replace(file_ext, f"_translated{file_ext}")

        # Run translation with real progress updates
        langgraph_pipeline(temp_input_path, temp_output_path, update_progress)

        # Remove input file after processing
        os.remove(temp_input_path)

        status.update(label="Translation Complete! 🎉", state="complete", expanded=False)

    # Display download button
    st.success("Your translated document is ready! 📂")
    with open(temp_output_path, "rb") as f:
        st.download_button("📥 Download Translated Document", f, file_name=f"translated{file_ext}")

    # Cleanup output file after download
    os.remove(temp_output_path)

    # Celebration animation
    st.balloons()
