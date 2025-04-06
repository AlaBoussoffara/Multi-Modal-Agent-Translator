"""
translator_agent.py

Ce module définit la classe `TranslatorAgent` pour traduire des paragraphes en utilisant un modèle de langage (LLM),
tout en préservant les métadonnées et la mise en forme. Il prend en charge la traduction par blocs avec une prise
en compte du contexte.

"""

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

class TranslatorAgent:
    """
    Agent de traduction utilisant un modèle de langage (LLM) pour traduire des paragraphes standardisés
    tout en préservant les métadonnées.

    Attributs :
        model : Le modèle de traduction (LLM) utilisé pour effectuer la traduction.
        target_language (str) : La langue cible pour la traduction.
        max_chunk_words (int) : Nombre maximum de mots autorisés par bloc de traduction.
    """
    def __init__(self, model, target_language='french', max_chunk_words=20):
        """
        Initialise le TranslatorAgent avec le modèle de traduction, la langue cible et le nombre maximum
        de mots par bloc.

        Args:
            model: Le modèle de traduction ou None pour effectuer une traduction fictive.
            target_language (str, optionnel): Langue cible. Par défaut : 'french'.
            max_chunk_words (int, optionnel): Nombre maximum de mots par bloc. Par défaut : 20.
        """
        self.model = model
        self.target_language = target_language
        self.max_chunk_words = max_chunk_words
        self._terminal_pbar = None  # Barre de progression pour la sortie terminal
        self._pbar_initialized = False  # Indique si la barre de progression a été initialisée

    def translate(self, paragraphs, progress_callback=None, terminal_progress=True):
        """
        Traduit une liste de paragraphes standardisés.

        Chaque paragraphe est divisé en blocs en fonction de la limite maximale de mots. La méthode traduit
        chaque bloc tout en tenant compte du contexte des blocs précédents et suivants, puis reconstruit le paragraphe.

        Args:
            paragraphs (list): Liste de paragraphes à traduire.
            progress_callback (callable, optionnel): Fonction de rappel pour mettre à jour la progression.
            terminal_progress (bool, optionnel): Affiche une barre de progression dans le terminal. Par défaut : True.

        Returns:
            list: Liste de paragraphes traduits avec leurs métadonnées préservées.
        """
        def split_text(text, max_words):
            """
            Divise un texte en blocs contenant un nombre maximum de mots.

            Args:
                text (str): Le texte à diviser.
                max_words (int): Nombre maximum de mots par bloc.

            Returns:
                list: Liste des blocs de texte.
            """
            words = text.split()
            return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

        def escape_curly_braces(text):
            """
            Échappe les accolades dans le texte pour éviter les problèmes de formatage.

            Args:
                text (str): Le texte d'entrée.

            Returns:
                str: Le texte avec les accolades échappées.
            """
            return text.replace("{", "{{").replace("}", "}}") if text else text

        translated_data = []
        previous_translated_chunk = ""

        # Compte le nombre total de blocs à traduire pour la barre de progression
        total_chunks = 0
        for p in paragraphs:
            if p["text"].strip():
                total_chunks += len(split_text(p["text"], self.max_chunk_words))
        completed_chunks = 0

        # Initialise la barre de progression dans le terminal si activée
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
                # Si le paragraphe est vide, l'ajouter tel quel
                translated_data.append(para)
                continue

            current_original = para["text"]
            text_chunks = split_text(current_original, self.max_chunk_words)
            translated_chunks = []

            # Extrait le contexte des paragraphes voisins
            previous_original_paragraph = " ".join(
                paragraphs[i - 1]["text"].split()[-20:]
            ) if i > 0 else ""
            next_original_paragraph = " ".join(
                paragraphs[i + 1]["text"].split()[:20]
            ) if i < len(paragraphs) - 1 else ""

            previous_original_paragraph = escape_curly_braces(previous_original_paragraph)
            next_original_paragraph = escape_curly_braces(next_original_paragraph)

            for j, chunk in enumerate(text_chunks):
                # Extrait le contexte des blocs voisins
                previous_original_chunk = " ".join(text_chunks[j - 1].split()[-20:]) if j > 0 else previous_original_paragraph
                next_original_chunk = " ".join(text_chunks[j + 1].split()[:20]) if j < len(text_chunks) - 1 else next_original_paragraph

                previous_original_chunk = escape_curly_braces(previous_original_chunk)
                next_original_chunk = escape_curly_braces(next_original_chunk)
                current_chunk = escape_curly_braces(chunk)
                prev_translated_escaped = escape_curly_braces(previous_translated_chunk)

                # Prépare le prompt pour le modèle LLM
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
                    # Traduction fictive : retourne le texte original
                    translated_text = chunk
                else:
                    chain = prompt_template | self.model | StrOutputParser()
                    translated_text = chain.invoke(input_dict)
                
                translated_chunks.append(translated_text)
                previous_translated_chunk = translated_text

                # Met à jour la progression
                completed_chunks += 1
                if pbar:
                    pbar.update(1)
                if progress_callback:
                    progress_percentage = int((completed_chunks / total_chunks) * 100)
                    progress_callback(progress_percentage)

            # Reconstruit le paragraphe avec le texte traduit tout en préservant les métadonnées
            translated_paragraph_text = " ".join(translated_chunks)
            translated_paragraph = para.copy()
            translated_paragraph["text"] = translated_paragraph_text
            translated_data.append(translated_paragraph)

        return translated_data