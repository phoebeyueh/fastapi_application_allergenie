import os
import re
import io
import json
import faiss
import boto3
import logging
import pandas as pd
import numpy as np
from urllib.parse import quote
from pydantic import BaseModel, Field
from io import StringIO, BytesIO
from typing import List, Dict
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import SagemakerEndpoint
from langchain_community.embeddings import SagemakerEndpointEmbeddings
from langchain_community.llms.sagemaker_endpoint import LLMContentHandler
from langchain_community.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain_community.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware
from sagemaker.predictor import Predictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up AWS and file paths
region = "us-east-1"
llm_endpoint_name = "llama3-allergenie"
embedding_endpoint_name = "embedding-allergenie"
sd_endpoint_name = "allergenie-model-txt2img-stabilityai-st-2024-11-16-18-48-29-504"

# Initialize the SageMaker runtime client
sagemaker_runtime_client = boto3.client('sagemaker-runtime',
                                        region_name= region)

# LLM: Define the Llama38BContentHandler class
class Llama38BContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": model_kwargs.get("max_new_tokens", 1000),
                "top_p": model_kwargs.get("top_p", 0.9),
                "temperature": model_kwargs.get("temperature", 0.8),
                "stop": ["<|eot_id|>"],
            },
        }
        input_str = json.dumps(payload)
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.decode("utf-8"))
        content = response_json["generated_text"].strip()
        return content

# LLM: content handler
content_handler = Llama38BContentHandler()

def invoke_endpoint(endpoint_name: str, prompt: str) -> str:
    try:
        model_kwargs = {"max_new_tokens": 1000, "top_p": 0.9, "temperature": 0.8}
        input_data = content_handler.transform_input(prompt, model_kwargs)

        response = sagemaker_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=input_data
        )

        output_data = response['Body'].read()
        transformed_output = content_handler.transform_output(output_data)

        return transformed_output

    except Exception as e:
        logger.error(f"Error invoking the endpoint: {str(e)}")
        return None

# Embeddings: Define the BGEContentHandlerV15 class
class BGEContentHandlerV15(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, text_inputs: List[str], model_kwargs: dict) -> bytes:
        input_str = json.dumps(
            {
                "text_inputs": text_inputs,
                **model_kwargs
            }
        )
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["embedding"]

# Embeddings: content handler
bge_content_handler = BGEContentHandlerV15()

sagemaker_embeddings = SagemakerEndpointEmbeddings(
    endpoint_name= embedding_endpoint_name,
    region_name=region,
    model_kwargs={"mode": "embedding"},
    content_handler=bge_content_handler,
)

def generate_image_from_title(title: str) -> str:
    logger.info(f"Generating image for '{title}'")

    try:
        # Initialize the SageMaker predictor
        predictor = Predictor(endpoint_name=sd_endpoint_name,region_name='us-east-1')
        # Send the payload to the SageMaker endpoint
        prompt = f"professional food photography of {title}, high resolution, appetizing, style plating, restaurant quality"
        response = predictor.predict(prompt, {"ContentType": "application/x-text", "Accept": "application/json"})
        response_json = json.loads(response)
        pil_image = Image.fromarray(np.uint8(response_json["generated_image"]))

        # current_directory = os.getcwd()
        # output_path = os.path.join(current_directory, f"{title}_image.jpg")

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)

        s3 = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'recipe-image-generator'
        formatted_title = quote(title)
        s3_key = f"images/{formatted_title}.jpg"

        # Upload directly from the buffer
        s3.upload_fileobj(buffer, bucket_name, s3_key, ExtraArgs={"ACL": "public-read"})
        logger.info(f"Image uploaded directly to S3 bucket '{bucket_name}' with key '{s3_key}'")

        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"

        # Return a clickable link
        return s3_url

    except Exception as e:
        logger.error(f"Error generating image for '{title}': {str(e)}")
        return f"Failed to generate image for '{title}'."

# Load documents from S3 bucket and create the FAISS index
def create_faiss_index() -> VectorStoreIndexWrapper:
    # Load CSV files from S3
    s3 = boto3.client('s3')
    bucket_name = 'hf-recipedata-csv'
    response = s3.list_objects_v2(Bucket=bucket_name)

    # Load each CSV file into a DataFrame and combine them
    dataframes = []
    for obj in response.get('Contents', []):
        if obj['Key'].endswith('.csv'):
            csv_obj = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
            body = csv_obj['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(body))
            dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    texts_from_csv = combined_df['combined_text'].tolist()

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=102,
    )
    document_chunks = text_splitter.split_documents(
        [Document(page_content=text) for text in texts_from_csv]
    )

    # Create FAISS index from document chunks
    vectorstore_faiss = FAISS.from_documents(
        document_chunks,
        sagemaker_embeddings
    )

    # Wrap the FAISS vector store with VectorStoreIndexWrapper
    wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)
    return wrapper_store_faiss

# Create the FAISS index
wrapper_store_faiss = create_faiss_index()

# Define the prompt template
prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant that generates recipes in a well-structured format.
Generate THREE (3) COMPLETELY DIFFERENT recipes based on the user's request.

Ensure that ingredients are listed only once in each recipe and avoid repeating any entries. The recipes should each include the following sections:
1. **Title** of the recipe
2. **Ingredients List** with measurements (avoid repeats)
3. **Instructions** in numbered steps for preparation

Return the recipes in the following format, ensuring each of the three recipes is structured exactly as follows:

**Recipe 1:**
**Title:**
Recipe title here
**Ingredients:**
- Ingredient 1
- Ingredient 2
- Ingredient 3
**Instructions:**
1. Step 1
2. Step 2
3. Step 3

**Recipe 2:**
**Title:**
Recipe title here
**Ingredients:**
- Ingredient 1
- Ingredient 2
- Ingredient 3
**Instructions:**
1. Step 1
2. Step 2
3. Step 3

**Recipe 3:**
**Title:**
Recipe title here
Ingredients:
- Ingredient 1
- Ingredient 2
- Ingredient 3
**Instructions:**
1. Step 1
2. Step 2
3. Step 3

Make sure the formatting follows the above structure exactly and that each recipe is completely unique.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["question"]
)

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecipeRequest(BaseModel):
    allergy: str = Field(..., description="Enter Your Allergy (e.g., gluten, dairy, treenuts)")
    dish_type: str = Field(..., description="Enter Meal Type (e.g., dessert, main course) or Cuisine Type (e.g., Italian, Japanese, Asian)")

@app.post("/generate-recipe/")
async def generate_recipe(request: RecipeRequest):
    allergy = request.allergy
    dish_type = request.dish_type
    question = f"I am allergic to {allergy}, can you give me three {dish_type} recipes free of {allergy}?"

    try:
        # Use the wrapper store as a retriever
        retriever = wrapper_store_faiss.vectorstore.as_retriever()

        # Retrieve relevant documents
        context_documents = retriever.get_relevant_documents(question)

        # Check if any documents were retrieved
        if not context_documents or not isinstance(context_documents, list):
            logger.warning("No documents found for the question.")
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        context = "\n".join([doc.page_content for doc in context_documents if hasattr(doc, 'page_content')])

        # Generate response using Llama3
        prompt = PROMPT.format(context = context, question =question)
        response = invoke_endpoint(llm_endpoint_name, prompt)

        if response is None:
            raise HTTPException(status_code=500, detail="Error generating recipe from LLM.")

        # Extract recipe titles using regex
        recipe_titles = re.findall(r"\*\*Title:\*\*\s*(.*)", response)
        if not recipe_titles:
            raise HTTPException(status_code=500, detail="Error extracting recipe titles.")

        images = {title: generate_image_from_title(title) for title in recipe_titles}

        return JSONResponse(content={
            "question": question,
            "recipe": response,
            "images": images,})

    except Exception as e:
        logger.error(f"Error generating recipe: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating recipe.")

# Run the app locally
if __name__ == '__main__':
    app.run(debug=True)
