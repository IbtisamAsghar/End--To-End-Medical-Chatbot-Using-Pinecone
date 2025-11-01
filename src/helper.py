from langchain.document_loaders import DirectoryLoader , PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings


# load and extract the text from the files 

def load_pdf_data (Data) :
    loader = DirectoryLoader(
        path = Data,
        glob = '*.pdf',
        loader_cls = PyPDFLoader
    )
    documents = loader.load()
    return documents


def filter_to_minimal_docs (documents : List[Document]) -> List[Document] :  
    """
    Filter the list of Document objects and give a new list of document that contains :
    1. Source in metadata
    2. Original Page_content
    """
    minimal_docs : List[Document] = []
    for doc in documents :
        src = doc.metadata.get('source') 
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata = {'source':src}
            )
        )
    return minimal_docs

# Splitting the documents into the chunks 

def split_documents_to_chunk(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 300,
        length_function = len
    )
    docs = text_splitter.split_documents(documents)
    return docs


# Download the embeding model from the hugging face
def download_embeddings ():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name
    )

    return embeddings