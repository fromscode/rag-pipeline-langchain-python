from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def ingest():
    data = TextLoader('data.txt')
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
    ).split_documents(data.load())

    return chunks


if __name__ == '__main__':
    chunks = ingest()
    for chunk in chunks:
        print(chunk.page_content)