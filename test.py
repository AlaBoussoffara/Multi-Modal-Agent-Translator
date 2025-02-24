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

class TranslatorAgent:
    def __init__(self, model, target_language='french'):
        self.model = model
        self.target_language = target_language

    def translate_text(self, text):
        if not text.strip():
            return text  # Avoid translating empty text
        
        prompt_template = ChatPromptTemplate([
            ("system", f"Translate the following text into {self.target_language}, maintaining the original document layout, font size, and color:"),
            ("user", "{text_to_translate}")
        ])

        input_dict = {"text_to_translate": text}
        chain = prompt_template | self.model | StrOutputParser()
        translated_text = chain.invoke(input_dict)
        return translated_text

    def translate_pdf(self, input_pdf, output_pdf):
        doc = fitz.open(input_pdf)
        translated_doc = fitz.open()

        for page in doc:
            text_blocks = page.get_text("dict")
            new_page = translated_doc.new_page(width=page.rect.width, height=page.rect.height)

            for block in text_blocks["blocks"]:
                for line in block["lines"]:
                    for span in line["spans"]:
                        x, y = span["origin"]
                        original_text = span["text"]
                        font_size = span["size"]
                        font_name = span["font"]
                        color = span["color"]
                        
                        translated_text = self.translate_text(original_text)
                        
                        new_page.insert_text(
                            (x, y), translated_text, fontsize=font_size, color=color
                        )

        translated_doc.save(output_pdf)
        translated_doc.close()
        print(f"Translated PDF saved as {output_pdf}")

# Initialize translator agent
translator = TranslatorAgent(llm, target_language="french")

# Translate a PDF file
translator.translate_pdf("input1.pdf", "translated.pdf")

