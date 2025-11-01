from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import TFIDFRetriever, ContextualCompressionRetriever
from langchain.document_transformers import EmbeddingsRedundantFilter, LongContextReorder
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from src.helper import download_embeddings , load_pdf_data

# =============================
# Build retriever pipeline
# =============================
def build_retriever(index_name="medical-chatbot", k=8):
    embeddings = download_embeddings()

    # Pinecone retriever
    docsearch = PineconeVectorStore.from_existing_index(
        embedding=embeddings,
        index_name=index_name,
    )
    retriever = docsearch.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "include_metadata": True},
    )
    # Filters + Reordering
    filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    reordering = LongContextReorder()
    pipeline = DocumentCompressorPipeline(transformers=[filter, reordering])

    # Final compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline,
        base_retriever=retriever,
        search_kwargs={"k": k, "include_metadata": True},
    )

    return compression_retriever
