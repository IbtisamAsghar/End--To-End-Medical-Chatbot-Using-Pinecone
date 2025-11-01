from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.prompt import medical_chat_prompt



def build_qa_chain(retriever):
    llm_model = ChatOpenAI(model="gpt-4o", streaming=True, temperature=0)
    parser = StrOutputParser()

    def docs_to_text(docs):
        return "\n\n".join([d.page_content for d in docs])
    
    qa_chain = (
        {"context": retriever | RunnableLambda(docs_to_text), "query": RunnablePassthrough()}
        | medical_chat_prompt
        | llm_model
        | parser
    )
    return qa_chain

