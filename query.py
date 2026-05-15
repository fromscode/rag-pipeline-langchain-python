from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

from vector_store import vector_store

import sys

def main():
    load_dotenv()
    query = " ".join(sys.argv[1:])

    store = vector_store()
    top_k = store.similarity_search(query, k=5)

    template = """
        You are a professional chatbot. You are asked a question / query, your job is to only use the below provided context and answer.
        If the below provided context cannot succesfully answwer the question, simply say "I don't know".

        Context:
        {context}

        Query:
        {query}
    """

    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=template
    )

    llm = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')


    lambda a: {"c": 'a', 's': ' '.join(a)}


    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(
        {
            "context": "\n".join([doc.page_content for doc in top_k]),
            "query": query
        }
    )

if __name__ == "__main__":
    print(main())