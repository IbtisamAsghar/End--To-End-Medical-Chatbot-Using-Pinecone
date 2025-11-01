from dotenv import load_dotenv
load_dotenv()
import os
from src.helper import load_pdf_data,filter_to_minimal_docs,split_documents_to_chunk,download_embeddings
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore


PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
os.environ['OPENAI_API_KEY']=OPENAI_API_KEY


extract_docs=load_pdf_data(r"C:\Users\PC\OneDrive\Desktop\End-To-End Medical ChatBot Using PineCone\End--To-End-Medical-Chatbot-Using-Pinecone\Data")
filtered_docs=filter_to_minimal_docs(extract_docs)
chunks_docs=split_documents_to_chunk(filtered_docs)
embeddings=download_embeddings()


pinecone_api=PINECONE_API_KEY
PC=Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"



# Creating the pinecone index but i already created during testing
# if not PC.has_index(index_name):
#     PC.create_index(
#         name=index_name,
#         dimension=384,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )
# index = PC.Index(index_name)



# load the existing index
docsearch2=PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name,
)