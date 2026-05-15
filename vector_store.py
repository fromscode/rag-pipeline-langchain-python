from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

from ingester import ingest

def vector_store():
    chunks = ingest()
    return FAISS.from_documents(chunks, embedding=FastEmbedEmbeddings())

if __name__ == '__main__':
    vector_store()