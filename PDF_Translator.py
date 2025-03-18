from collections import Counter
import numpy as np
import os
from PIL import Image
import fitz  # PyMuPDF
import boto3
import botocore
from langchain_aws import ChatBedrock
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize AWS Bedrock client
client_config = botocore.config.Config(max_pool_connections=100)
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1", config=client_config)

# Initialize AI model for translation
llm = ChatBedrock(
    client=None,
    model_id="us.meta.llama3-3-70b-instruct-v1:0",
    region_name="us-west-2",
    model_kwargs={"temperature": 0},
)


class PDFTranslatorAgent:
    """
    Agent for extracting, translating, and re-inserting text in PDF documents.

    Methods:
      - translate_pdf: Translates paragraphs preserving context.
      - extract_paragraphs: Extracts structured paragraph data from a PDF.
      - get_dominant_font_properties: Determines common font properties for a paragraph.
      - get_background_color_from_bbox: Samples a bounding box to estimate its background color,
                                           excluding a given font color.
      - create_pdf_from_translated_data: Creates a new PDF by redacting original text areas
                                           and inserting translated text.
    """

    def __init__(self, model, target_language='french', max_chunk_words=20):
        """
        Initializes the PDFTranslatorAgent.

        Parameters:
            model: The LLM model used for translation.
            target_language (str): The language into which text is translated.
            max_chunk_words (int): Maximum number of words per translation chunk.
        """
        self.model = model
        self.target_language = target_language
        self.max_chunk_words = max_chunk_words

    def translate_pdf(self, structured_data):
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
                previous_translated_chunk = escape_curly_braces(previous_translated_chunk)

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
Previous Translated Chunk: "{previous_translated_chunk}"
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

            # Concatenate all translated chunks to form the full paragraph
            translated_paragraph_text = " ".join(translated_chunks)
            translated_paragraph = paragraph.copy()
            translated_paragraph["text"] = translated_paragraph_text
            translated_data.append(translated_paragraph)

        return translated_data

    def extract_paragraphs(self, pdf_path, tolerance_factor=2.5):
        """
        Extracts paragraphs from a PDF while preserving structure and layout.

        The method processes text blocks from the PDF and groups them into paragraphs based on
        spacing, font properties, and bounding box coordinates.

        Parameters:
            pdf_path (str): Path to the input PDF.
            tolerance_factor (float): Factor to adjust spacing threshold for paragraph separation.

        Returns:
            list of dicts: Each dict represents a paragraph with keys such as "text", "bbox",
                           "page", "font", "size", "color", "bold", "italic", "char_spacing", and "spacing".
        """
        doc = fitz.open(pdf_path)
        structured_data = []

        for page_num, page in enumerate(doc):
            raw_blocks = []
            # Extract text spans from each block/line in the page
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue

                    text = " ".join(span["text"] for span in spans).strip()
                    if not text:
                        continue

                    font = spans[0]["font"]
                    font_size = spans[0]["size"]
                    color = spans[0]["color"]
                    flags = spans[0]["flags"]
                    char_spacing = spans[0].get("char_spacing", 0)

                    is_bold = bool(flags & 8)
                    is_italic = bool(flags & 4)
                    bbox = line["bbox"]

                    raw_blocks.append({
                        "text": text,
                        "bbox": bbox,
                        "page": page_num,
                        "font": font,
                        "size": font_size,
                        "color": color,
                        "bold": is_bold,
                        "italic": is_italic,
                        "char_spacing": char_spacing,
                    })

            # Calculate spacing threshold based on the minimum gap between blocks
            line_gaps = [raw_blocks[i+1]["bbox"][1] - raw_blocks[i]["bbox"][3]
                         for i in range(len(raw_blocks) - 1)]
            auto_threshold = (min(line_gaps) * tolerance_factor) if line_gaps else 10

            paragraphs = []
            current_paragraph = None
            paragraph_properties = []

            for i, block in enumerate(raw_blocks):
                if i == 0:
                    current_paragraph = {
                        "text": block["text"],
                        "bbox": block["bbox"],
                        "page": page_num,
                        "font": block["font"],
                        "size": block["size"],
                        "color": block["color"],
                        "bold": block["bold"],
                        "italic": block["italic"],
                        "char_spacing": block["char_spacing"],
                        "spacing": 1
                    }
                    paragraph_properties.append(block)
                    continue

                prev_bottom = current_paragraph["bbox"][3]
                curr_top = block["bbox"][1]
                vertical_gap = abs(prev_bottom - curr_top)

                # If the font or size changes or the gap is too large, start a new paragraph.
                if ((current_paragraph["font"] != block["font"]) or
                    (abs(current_paragraph["size"] - block["size"]) > 1) or
                    (vertical_gap > auto_threshold)):
                    dominant_properties = self.get_dominant_font_properties(paragraph_properties)
                    current_paragraph.update(dominant_properties)
                    paragraphs.append(current_paragraph)
                    current_paragraph = {
                        "text": block["text"],
                        "bbox": block["bbox"],
                        "page": page_num,
                        "font": block["font"],
                        "size": block["size"],
                        "color": block["color"],
                        "bold": block["bold"],
                        "italic": block["italic"],
                        "char_spacing": block["char_spacing"],
                        "spacing": 1
                    }
                    paragraph_properties = [block]
                else:
                    # Append text and update the bounding box of the current paragraph.
                    current_paragraph["spacing"] = max(1.75 * vertical_gap / block["size"],
                                                      current_paragraph["spacing"])
                    current_paragraph["text"] += " " + block["text"]
                    current_paragraph["bbox"] = (
                        min(current_paragraph["bbox"][0], block["bbox"][0]),
                        current_paragraph["bbox"][1],
                        max(block["bbox"][2], current_paragraph["bbox"][2]),
                        block["bbox"][3]
                    )
                    paragraph_properties.append(block)

            if current_paragraph:
                dominant_properties = self.get_dominant_font_properties(paragraph_properties)
                current_paragraph.update(dominant_properties)
                paragraphs.append(current_paragraph)

            structured_data.extend(paragraphs)

        return structured_data

    def get_dominant_font_properties(self, paragraph_blocks):
        """
        Determines the most frequent font properties in a paragraph.

        Parameters:
            paragraph_blocks (list of dicts): Blocks that form a paragraph.

        Returns:
            dict: Dominant properties including "font", "size", "color", "bold", "italic", and "char_spacing".
        """
        def most_frequent(items):
            return Counter(items).most_common(1)[0][0] if items else None

        fonts = [block["font"] for block in paragraph_blocks]
        sizes = [block["size"] for block in paragraph_blocks]
        colors = [block["color"] for block in paragraph_blocks]
        bold_flags = [block["bold"] for block in paragraph_blocks]
        italic_flags = [block["italic"] for block in paragraph_blocks]
        char_spacings = [block["char_spacing"] for block in paragraph_blocks]

        return {
            "font": most_frequent(fonts),
            "size": min(sizes),
            "color": most_frequent(colors),
            "bold": most_frequent(bold_flags),
            "italic": most_frequent(italic_flags),
            "char_spacing": np.mean(char_spacings)
        }

    def get_background_color_from_bbox(self, pdf_path, page_number, bbox, zoom=2, border_width=5, exclude_color=None, tolerance=20):
        """
        Determines the background color of a given bounding box by sampling border pixels.
        Excludes pixels close to the provided exclude_color (e.g., font color) within the tolerance.

        Parameters:
            pdf_path (str): Path to the PDF file.
            page_number (int): The 0-indexed page number.
            bbox (list/tuple): Bounding box [x0, y0, x1, y1] in PDF points.
            zoom (int): Zoom factor for higher resolution.
            border_width (int): Thickness (in pixels) of the border region.
            exclude_color (tuple, optional): Normalized (R, G, B) color to exclude.
            tolerance (int): Tolerance (0–255) for excluding similar pixels.

        Returns:
            tuple: (R, G, B) normalized to 0–1 representing the background color.
        """
        # Render the page at increased resolution
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Convert bbox from PDF points to pixel coordinates
        x0, y0, x1, y1 = [int(coord * zoom) for coord in bbox]
        cropped = img.crop((x0, y0, x1, y1))
        arr = np.array(cropped)  # shape: (height, width, 3)
        height, width, _ = arr.shape

        # Sample border pixels (top, bottom, left, right)
        border_pixels = []
        border_pixels.append(arr[0:border_width, :, :].reshape(-1, 3))       # Top border
        border_pixels.append(arr[-border_width:, :, :].reshape(-1, 3))         # Bottom border
        if height > 2 * border_width:
            border_pixels.append(arr[border_width:height-border_width, 0:border_width, :].reshape(-1, 3))  # Left border
            border_pixels.append(arr[border_width:height-border_width, -border_width:, :].reshape(-1, 3))      # Right border

        border_pixels = np.concatenate(border_pixels, axis=0)
        pixel_tuples = [tuple(pixel) for pixel in border_pixels]

        # Exclude pixels near the provided font color if given
        if exclude_color is not None:
            target = tuple(int(c * 255) for c in exclude_color)
            filtered_pixels = [p for p in pixel_tuples if not (
                abs(p[0] - target[0]) < tolerance and
                abs(p[1] - target[1]) < tolerance and
                abs(p[2] - target[2]) < tolerance
            )]
            mode_color = (Counter(filtered_pixels).most_common(1)[0][0]
                          if filtered_pixels else Counter(pixel_tuples).most_common(1)[0][0])
        else:
            mode_color = Counter(pixel_tuples).most_common(1)[0][0]

        normalized_color = tuple(c / 255 for c in mode_color)
        doc.close()
        return normalized_color

    def create_pdf_from_translated_data(self, translated_data, original_pdf, output_pdf):
        """
        Creates a new PDF by duplicating the original, redacting text in extracted areas,
        and inserting translated text into those areas.

        Parameters:
            translated_data (list of dicts): Structured data with keys "page", "bbox", "text", "font",
                                               "size", "color", "bold", "italic", "spacing".
            original_pdf (str): Path to the original PDF.
            output_pdf (str): Path to save the modified PDF.
        """
        # Open the original PDF
        doc = fitz.open(original_pdf)

        # === STEP 1: Redact original text ===
        for para in translated_data:
            page_num = para["page"]
            bbox = para["bbox"]
            # Slightly shrink the bbox to avoid overlapping boundaries
            bbox = tuple([bbox[0] + 2, bbox[1] + 2, bbox[2] - 2, bbox[3] - 2]) if (bbox[0] + 2 < bbox[2] - 2 and bbox[1] + 2 < bbox[3] - 2) else bbox
            page = doc[page_num]
            redaction_rect = fitz.Rect(bbox)
            # Convert font color from int to normalized RGB to use as exclude_color
            color_int = para["color"]
            r = (color_int >> 16) & 0xFF
            g = (color_int >> 8) & 0xFF
            b = color_int & 0xFF
            font_color = (r / 255, g / 255, b / 255)
            # Determine fill color by sampling background, excluding the font color
            fill_color = self.get_background_color_from_bbox(original_pdf, page_num, bbox, zoom=2, exclude_color=font_color, tolerance=20)
            page.add_redact_annot(redaction_rect, fill=fill_color)

        for page in doc:
            page.apply_redactions()

        # === STEP 2: Insert translated text ===
        for para in translated_data:
            page_num = para["page"]
            page = doc[page_num]
            bbox = para["bbox"]
            # Adjust bbox to ensure text fits well
            bbox = tuple([bbox[0] + 2, bbox[1] + 2, bbox[2] - 2, bbox[3] - 2]) if (bbox[0] + 2 < bbox[2] - 2 and bbox[1] + 2 < bbox[3] - 2) else bbox
            x0, y0, x1, y1 = bbox
            text = para["text"]
            font = para["font"]
            font_size = para["size"]
            color_int = para["color"]
            is_bold = para.get("bold", False)
            is_italic = para.get("italic", False)
            spacing = para.get("spacing", 1)

            if not text.strip():
                continue

            # Convert color int to normalized RGB tuple
            r = (color_int >> 16) & 0xFF
            g = (color_int >> 8) & 0xFF
            b = color_int & 0xFF
            color_rgb = (r / 255, g / 255, b / 255)

            # Choose valid font, defaulting to Times variants if needed
            valid_fonts = ["Helvetica", "Courier"]
            if font not in valid_fonts:
                if is_bold and is_italic:
                    font = "Times-BoldItalic"
                elif is_bold:
                    font = "Times-Bold"
                elif is_italic:
                    font = "Times-Italic"
                else:
                    font = "Times-Roman"
            else:
                if is_bold and is_italic:
                    font += "-BoldOblique"
                elif is_bold:
                    font += "-Bold"
                elif is_italic:
                    font += "-Oblique"

            # Set padding for the textbox
            padding = 20
            text_box_rect = fitz.Rect(x0 + padding / 4, y0, x1 + padding, y1 + padding)

            page.insert_textbox(
                text_box_rect,
                text,
                fontsize=font_size * 0.8,
                fontname=font,
                color=color_rgb,
                lineheight=spacing,
                align=0  # Left-aligned; adjust as needed
            )

        # === STEP 3: Save the modified PDF ===
        doc.save(output_pdf)
        doc.close()
        print(f"✅ New PDF created from original '{original_pdf}' with translated text saved as '{output_pdf}'")


# ==================== Main Processing Loop ====================
# Define input and output directories
input_dir = "documents à traduire"
output_dir = "documents traduits"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each PDF in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".pdf"):
        input_pdf_path = os.path.join(input_dir, filename)
        # Construct output path: replace extension with '_translated.pdf'
        output_pdf_path = os.path.join(output_dir, filename.rsplit(".", 1)[0] + "_translated.pdf")

        # Determine target language based on filename
        target_language = "french" if "EN" in filename else "english"

        # Initialize translator agent with determined target language
        translator = PDFTranslatorAgent(llm, target_language=target_language)

        # Extract paragraphs from the PDF
        pdf_data = translator.extract_paragraphs(input_pdf_path)
        # Translate the extracted paragraphs
        pdf_translated = translator.translate_pdf(pdf_data)
        # Create a new PDF with the translated text inserted
        translator.create_pdf_from_translated_data(pdf_translated, input_pdf_path, output_pdf_path)