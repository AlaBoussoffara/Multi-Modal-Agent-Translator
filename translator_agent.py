import boto3
import botocore
import chainlit as cl
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from typing import Any, Dict, List
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn



# intiatilization of the bedrock client
client_config = botocore.config.Config(max_pool_connections=100)
bedrock_client = boto3.client(
            "bedrock-runtime", region_name="us-east-1", config=client_config
)

# intialization of the chat model
llm = ChatBedrock(
            client=None,
            model_id="us.meta.llama3-3-70b-instruct-v1:0",
            region_name="us-west-2",
            model_kwargs={"temperature":0},
        )
# intialized the chat prompt
embedding = BedrockEmbeddings(
            client=bedrock_client,
            model_id="amazon.titan-embed-text-v2:0",
            region_name="us-east-1",
        )







class TranslatorAgent:
    """
        this class is an example of Translation agent that uses the bedrock model to 
        translate text to a target language.
        It support normal text and JSON object translation.
    """
    def __init__(self, model=llm, target_language: str='french'):
        self.model = model
        self.target_language = target_language

    def prompt_template_json_to_json(self):
        return """
            # TRANSLATION RULES
            Your and agent desitinated to translate text to a target language.
            - Do the translation by using the appropriate term in Tender Offer document
            - Ensure the translation is precise and conveys the exact meaning of the original document.
            - Maintain the original formatting and structure of the document to ensure clarity and consistency.
            - Ensure the translation complies with any specific requirements mentioned in the tender documents.

            # INSTRUCTION
            - You will be asked to translate the a \`\`\`JSON\`\`\`\` object to the target language. 
                respect the \`\`\`JSON\`\`\`\`  structure and the special characters.
            - The target language will be mentioned in the prompt.
            - Respect the special characters and formatting of the text.
            - Maintain the original formatting and structure of the document to ensure clarity and consistency.
            - Ensure the translation complies with any specific requirements mentioned in the tender documents
            - Do not add any additional information to the translation. Only translate the text provided. 
            - Do not comment on the text or provide any additional information.
            - If the content is in the target language, do not translate it back to english.

            # TRARGET LANGUAGE
            ----------------------------------------------------------------
            {target_language}
            ----------------------------------------------------------------

            # TEXT TO TRANSLATE
            ----------------------------------------------------------------
            {text_to_translate}
            ----------------------------------------------------------------
            
            # FINAL TRANSLATION
            - Do not add introduction in your response or any additional information, 
            no phrase like 'Voici', 'Voila', 'I think', 'I believe', 'I suggest', etc. 'Voici la traduction' or 'Voilà' are also not allowed.

            # TASK
            Translate the content \`\`\`JSON\`\`\`\`  object in english into  the target language, 
            while preserving the \`\`\`JSON\`\`\`\` object structure and special characters.

        """

    def prompt_template_text_to_text(self):
        return """
# TRANSLATION RULES
You are an agent designated to translate text into a target language.
- Ensure the translation is precise and conveys the exact meaning of the text.
- A good translation is accurate and precise.
- The field of translation concerns the "tender offer" document.

# INSTRUCTION

- The target language will be mentioned in the prompt.
- Respect the special characters and formatting of the text.
- If the text contains incorrectly encoded special characters (e.g., \t, \x06, etc.), ensure the final output is properly formatted and readable.
- Maintain the original formatting and structure of the document to ensure clarity and consistency.
- Ensure the translation complies with any specific requirements mentioned in the tender documents.
- Do not add any additional information to the translation. Only translate the text provided.
- Do not comment on the text or provide any additional information.
- If the content is already in the target language, do not translate it back to English.
- If the text is just a simple string with no actual content to translate (e.g., ": "), return the same string without modification.

# TARGET LANGUAGE
----------------------------------------------------------------
{target_language}
----------------------------------------------------------------

# TEXT TO TRANSLATE
----------------------------------------------------------------
{text_to_translate}
----------------------------------------------------------------

# FINAL TRANSLATION
- Do not add an introduction in your response or any additional information, 
  no phrases like "Voici", "Voilà", "I think", "I believe", "I suggest", etc. 
  Phrases such as "Voici la traduction" or "Voilà" are also not allowed.
- Perform a direct translation of the text into the target language.
- If special characters are misencoded, ensure they are properly interpreted and formatted.
- If the text is only a simple string with no actual content to translate, return it unchanged.

# TASK
Translate the content in English into the target language.
        """
    
    def prompt_template_html_to_html(self):
        return """# TRANSLATION RULES
You are an agent designated to translate text within an HTML document to a target language while maintaining the exact structure and formatting.

Ensure the translation is precise and conveys the exact meaning of the text.
Do not modify, remove, or add any HTML tags, attributes, or special characters.
Do not translate content that appears inside HTML tags (e.g., class names, IDs, attributes like href, src, alt, title).
Preserve the indentation, spacing, and line breaks exactly as in the original document.
If the content is already in the target language, do not translate it.

# INSTRUCTIONS
The target language will be specified in the prompt.
Only translate the text content, not the HTML structure.
Do not add any comments, explanations, or additional information.
Ensure the translation complies with any domain-specific requirements.

# TARGET LANGUAGE
{target_language}

# HTML CONTENT TO TRANSLATE
{text_to_translate}

# FINAL TRANSLATION
Do not add introductions like "Here is the translation," "Voila," etc.
Maintain the exact HTML structure and formatting while translating only the visible text content.

# TASK
Translate the textual content within the HTML while ensuring the integrity of the HTML structure remains unchanged.
"""

    def translate(self, content, parser=StrOutputParser()) -> str:
        
        """
        Translate the given content to the target language.

        Parameters:
        content (str or dict or list): The text or data to be translated. If a dictionary is provided, it will be converted to a list.
        parser (StrOutputParser, optional): An instance of StrOutputParser to process the output. Defaults to StrOutputParser().

        Returns:
        str: The translated text in the target language.

        If the content is empty, an empty string is returned.
        """
        if content != "":
            print("Translating the text to the target language ...")
            if isinstance(content, dict):
                content = [content]
            
            if "<p>" in content:
                print("HTML detected")
                prompt_template = self.prompt_template_html_to_html()
            elif isinstance(content, list):
                prompt_template = self.prompt_template_json_to_json()
            else:
                prompt_template = self.prompt_template_text_to_text()
            
            template = ChatPromptTemplate([
                ("system", prompt_template), 
                ("user", "{text_to_translate}")
            ]) 
            input_dict = {
                "target_language": self.target_language,
                "text_to_translate": content
            }

            chain = template | self.model | parser
            final_answer = chain.invoke(input_dict)
            return final_answer
        return ""
    

# Test the agent ========================================================
if __name__ == "__main__":
  translatior_agent = TranslatorAgent(llm, 'french')


  translatior_agent.translate(
  """
      Benjamin F. McAdoo (1920–1981) was an American architect mainly\n
      active in the Seattle area. Born in Pasadena, California, he was\n
      inspired to study architecture by a mechanical-drawing class\n
      and the work of Paul R. Williams.
  """)