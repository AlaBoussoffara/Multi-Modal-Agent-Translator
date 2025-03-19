# translator_agent.py

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm  # <-- NEW import for progress bar

class TranslatorAgent:
    def __init__(self, model, target_language='french', max_chunk_words=20):
        self.model = model
        self.target_language = target_language
        self.max_chunk_words = max_chunk_words  # Maximum words per translation chunk

    def translate(self, structured_data):
        """
        Translates the text of each paragraph using context from surrounding chunks.

        The method:
          - Splits long paragraphs into smaller chunks.
          - Uses context from previous and next chunks.
          - Invokes the LLM to generate a coherent translation that concatenates naturally.

        Parameters:
            structured_data (list of dicts): List of paragraph data containing keys like
                                               "text", "bbox", "font", "size", "color", etc.

        Returns:
            list of dicts: Updated structured data with the "text" field replaced by its translation.
        """
        translated_data = []
        previous_translated_chunk = ""  # To maintain sentence continuity

        def split_text(text, max_words):
            """Splits text into chunks with at most max_words words."""
            words = text.split()
            return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

        def escape_curly_braces(text):
            """Escapes curly braces to avoid formatting errors in the prompt."""
            return text.replace("{", "{{").replace("}", "}}") if text else text

        # Count total chunks to show progress
        total_chunks = 0
        for paragraph in structured_data:
            if paragraph["text"].strip():
                text_chunks = split_text(paragraph["text"], self.max_chunk_words)
                total_chunks += len(text_chunks)

        # Use a tqdm progress bar to track chunk translation
        with tqdm(total=total_chunks, desc="Translating", unit="chunk") as pbar:

            for i, paragraph in enumerate(structured_data):
                if not paragraph["text"].strip():
                    translated_data.append(paragraph)
                    continue

                current_original = paragraph["text"]
                text_chunks = split_text(current_original, self.max_chunk_words)
                translated_chunks = []

                # Extract context: previous and next paragraph (20 words) if available
                previous_original_paragraph = " ".join(
                    structured_data[i - 1]["text"].split()[-20:]
                ) if i > 0 else ""
                next_original_paragraph = " ".join(
                    structured_data[i + 1]["text"].split()[:20]
                ) if i < len(structured_data) - 1 else ""

                previous_original_paragraph = escape_curly_braces(previous_original_paragraph)
                next_original_paragraph = escape_curly_braces(next_original_paragraph)

                for j, chunk in enumerate(text_chunks):
                    # Get context from within the same paragraph
                    previous_original_chunk = " ".join(text_chunks[j - 1].split()[-20:]) if j > 0 else previous_original_paragraph
                    next_original_chunk = " ".join(text_chunks[j + 1].split()[:20]) if j < len(text_chunks) - 1 else next_original_paragraph

                    # Escape curly braces in each context chunk
                    previous_original_chunk = escape_curly_braces(previous_original_chunk)
                    next_original_chunk = escape_curly_braces(next_original_chunk)
                    current_chunk = escape_curly_braces(chunk)
                    prev_translated_escaped = escape_curly_braces(previous_translated_chunk)

                    # Construct prompt for LLM
                    prompt_template = ChatPromptTemplate([
                        ("system", f"""
Predict the continuation of the translation into {self.target_language} for the current original chunk.
Use the context so that the result concatenated with the previous translated chunk forms a coherent sentence.
                    
IMPORTANT:
- Keep the approximate word count equal or lower.
- Detect if the chunk is part of a larger sentence and adjust accordingly.
- Use the last 20 words of the previous chunk and the first 20 words of the next chunk for context.
- If no logical continuation is possible, translate the chunk as a standalone sentence.
- Preserve non-translatable elements (links, emails, phone numbers, names, specialized terms) exactly.
- Preserve the original formatting (bullet points, numbered lists, headings, indentations).

CONTEXT:
Previous Translated Chunk: "{prev_translated_escaped}"
Previous Original Chunk (last 20 words): "{previous_original_chunk}"
Current Chunk: "{current_chunk}"
Next Original Chunk (first 20 words): "{next_original_chunk}"

OUTPUT ONLY the translated version of the current chunk.
"""),
                        ("user", "{text_to_translate}")
                    ])

                    input_dict = {"text_to_translate": chunk}
                    chain = prompt_template | self.model | StrOutputParser()
                    translated_text = chain.invoke(input_dict)
                    translated_chunks.append(translated_text)
                    previous_translated_chunk = translated_text

                    # Update the progress bar
                    pbar.update(1)

                # Concatenate all translated chunks to form the full paragraph
                translated_paragraph_text = " ".join(translated_chunks)
                translated_paragraph = paragraph.copy()
                translated_paragraph["text"] = translated_paragraph_text
                translated_data.append(translated_paragraph)

        return translated_data
