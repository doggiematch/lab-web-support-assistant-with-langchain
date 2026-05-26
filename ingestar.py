from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hf_client import HuggingFaceInferenceEmbeddings

loader = TextLoader("politicas.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceInferenceEmbeddings()
vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

print(f"Indexados {len(chunks)} chunks")
