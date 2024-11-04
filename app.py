import os
import json
import boto3
import faiss
import logging
import pandas as pd
from pydantic import BaseModel, Field
from io import StringIO
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up AWS and file paths
# You can also create a user role with the following policies attached on your account: AmazonS3FullAccess, AmazonS3ReadOnlyAccess, AmazonSageMakerFullAccess, AWSLambda_FullAccess
region = "us-east-1"
llm_endpoint_name = "meta-textgeneration-llama-3-8b-instruct-2024-11-04-14-20-36-390"
embedding_endpoint_name = "hf-sentencesimilarity-bge-large-en-v1-5-2024-11-04-14-30-07-790"

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
                "max_new_tokens": model_kwargs.get("max_new_tokens", 1024),
                "top_p": model_kwargs.get("top_p", 0.9),
                "temperature": model_kwargs.get("temperature", 0.7),
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
        model_kwargs = {"max_new_tokens": 2048, "top_p": 0.9, "temperature": 0.7}
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
        chunk_size=2048,
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
Please generate a recipe.
Ensure that ingredients are listed only once and avoid repeating any entries.
The recipe should include the following sections:
1. Title of the recipe
2. Ingredients List with measurements (avoid repeats)
3. Instructions in numbered steps for preparation

Return the result in the following format:

Title:
Recipe title here

Ingredients:
- Ingredient 1
- Ingredient 2
- Ingredient 3

Instructions:
1. Step 1
2. Step 2
3. Step 3

Make sure the formatting follows the above structure.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
#### Context ####
{context}
#### End of Context ####

Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
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

app.mount("/static", StaticFiles(directory="static"), name="static")

class RecipeRequest(BaseModel):
    allergy: str = Field(..., description="Enter Your Allergy (e.g., gluten, dairy, treenuts)")
    dish_type: str = Field(..., description="Enter Meal Type (e.g., dessert, main course) or Cuisine Type (e.g., Italian, Japanese, Asian)")

@app.get("/")
async def get_html():
    return FileResponse("static/index.html")

@app.post("/generate-recipe/")
async def generate_recipe(request: RecipeRequest):
    allergy = request.allergy
    dish_type = request.dish_type
    question = f"I have a {allergy} allergy, and I would like to make a {dish_type} dish. Generate an allergy-free recipe for me."

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
        prompt = PROMPT.format(context=context, question=question)
        response = invoke_endpoint(llm_endpoint_name, prompt)

        if response is None:
            raise HTTPException(status_code=500, detail="Error generating recipe from LLM.")

        return {
            "question": question,
            "recipe": response
        }

    except Exception as e:
        logger.error(f"Error generating recipe: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating recipe.")

# Run the app locally
if __name__ == '__main__':
    app.run(debug=True)
