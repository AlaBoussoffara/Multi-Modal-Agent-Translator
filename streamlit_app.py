import streamlit as st
import os
import tempfile
from mmat_langgraph_version import langgraph_pipeline  # Import your model

# Streamlit App Configuration
st.set_page_config(page_title="Document Translator", page_icon="🌍")
st.title(":rainbow[Document Translator] 📄🌍")
st.write("Upload a document, and get a translated version while keeping the layout intact.")

uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx"])

if uploaded_file:
    file_name = uploaded_file.name
    file_ext = os.path.splitext(file_name)[-1]

    with st.status("Translating... Please wait! ⏳", expanded=False) as status:
        # Create temporary input file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_input:
            temp_input.write(uploaded_file.read())
            temp_input_path = temp_input.name

        # Create temporary output file path
        temp_output_path = temp_input_path.replace(file_ext, f"_translated{file_ext}")

        # Call your translation model
        langgraph_pipeline(temp_input_path, temp_output_path)

        # Remove input file after processing
        os.remove(temp_input_path)

        status.update(label="Translation Complete! 🎉", state="complete", expanded=False)

    st.success("Your translated document is ready! 📂")
    with open(temp_output_path, "rb") as f:
        st.download_button("📥 Download Translated Document", f, file_name=f"translated{file_ext}")

    # Cleanup output file after download
    os.remove(temp_output_path)

st.balloons()
