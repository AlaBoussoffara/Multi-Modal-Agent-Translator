
"""
translator_agent.py

This module defines the `TranslatorAgent` class for translating paragraphs using a language model (LLM) 
while preserving metadata and formatting. It supports chunk-based translation with contextual awareness.

"""

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

class TranslatorAgent:
    """
    Translation agent that uses a language model (LLM) to translate standardized paragraphs while preserving metadata.

    Attributes:
        model: The translation model (LLM) used to perform the translation.
        target_language (str): The target language for the translation.
        max_chunk_words (int): Maximum number of words allowed in each translation chunk.
    """
    def __init__(self, model, target_language='french', max_chunk_words=20):
        """
        Initialize the TranslatorAgent with the translation model, target language, and maximum chunk word count.

        Args:
            model: The translation model or None to perform a dummy translation.
            target_language (str, optional): Target language. Default: 'french'.
            max_chunk_words (int, optional): Maximum number of words per chunk. Default: 20.
        """
        self.model = model
        self.target_language = target_language
        self.max_chunk_words = max_chunk_words
        self._terminal_pbar = None  # Instance-level progress bar for terminal output
        self._pbar_initialized = False  # Flag to track progress bar initialization

    def translate(self, paragraphs, progress_callback=None, terminal_progress=True):
        """
        Translate a list of standardized paragraphs.

        Each paragraph is split into chunks based on the max word limit. The method translates each chunk
        while considering the context from previous and next chunks, and then reconstructs the paragraph.
        """
        def split_text(text, max_words):
            """
            Split a given text into chunks with a maximum number of words.
        
            Args:
                text (str): The text to split.
                max_words (int): Maximum number of words per chunk.
        
            Returns:
                list: List of text chunks.
            """
            words = text.split()
            return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

        def escape_curly_braces(text):
            """
            Escape curly braces in the given text to avoid formatting issues.
        
            Args:
                text (str): The input text.
        
            Returns:
                str: The text with curly braces escaped.
            """
            return text.replace("{", "{{").replace("}", "}}") if text else text
        
        translated_data = []
        previous_translated_chunk = ""

        # Count total chunks across all paragraphs for progress reporting
        total_chunks = 0
        for p in paragraphs:
            if p["text"].strip():
                total_chunks += len(split_text(p["text"], self.max_chunk_words))
        completed_chunks = 0

        # Initialize terminal progress bar if enabled
        if terminal_progress:
            if not self._pbar_initialized:
                self._terminal_pbar = tqdm(total=total_chunks, desc="Translating", unit="chunk")
                self._pbar_initialized = True
            else:
                self._terminal_pbar.reset(total=total_chunks)
            pbar = self._terminal_pbar
        else:
            pbar = None

        for i, para in enumerate(paragraphs):
            if not para["text"].strip():
                translated_data.append(para)
                continue

            current_original = para["text"]
            text_chunks = split_text(current_original, self.max_chunk_words)
            translated_chunks = []

            # Extract context from neighboring paragraphs
            previous_original_paragraph = " ".join(
                paragraphs[i - 1]["text"].split()[-20:]
            ) if i > 0 else ""
            next_original_paragraph = " ".join(
                paragraphs[i + 1]["text"].split()[:20]
            ) if i < len(paragraphs) - 1 else ""

            previous_original_paragraph = escape_curly_braces(previous_original_paragraph)
            next_original_paragraph = escape_curly_braces(next_original_paragraph)

            for j, chunk in enumerate(text_chunks):
                previous_original_chunk = " ".join(text_chunks[j - 1].split()[-20:]) if j > 0 else previous_original_paragraph
                next_original_chunk = " ".join(text_chunks[j + 1].split()[:20]) if j < len(text_chunks) - 1 else next_original_paragraph

                previous_original_chunk = escape_curly_braces(previous_original_chunk)
                next_original_chunk = escape_curly_braces(next_original_chunk)
                current_chunk = escape_curly_braces(chunk)
                prev_translated_escaped = escape_curly_braces(previous_translated_chunk)

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

                if self.model is None:
                    # Dummy translation: return the original chunk
                    # This is a fallback mechanism when no translation model is provided.
                    translated_text = chunk
                else:
                    chain = prompt_template | self.model | StrOutputParser()
                    translated_text = chain.invoke(input_dict)
                
                translated_chunks.append(translated_text)
                previous_translated_chunk = translated_text

                completed_chunks += 1
                if pbar:
                    pbar.update(1)
                if progress_callback:
                    progress_percentage = int((completed_chunks / total_chunks) * 100)
                    progress_callback(progress_percentage)

            # Reconstruct the paragraph with translated text while preserving metadata
            translated_paragraph_text = " ".join(translated_chunks)
            translated_paragraph = para.copy()
            translated_paragraph["text"] = translated_paragraph_text
            translated_data.append(translated_paragraph)

        return translated_data