import os

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Get the absolute path of the project directory
working_dir = os.path.abspath("biochem_chat")

# Define paths using the working directory
data_path = os.path.join(working_dir, "data")  # Ensure "data" is inside "biochem_chat"
vector_db_path = os.path.join(working_dir, "vector_db_dir")  # Ensure persistence in "biochem_chat"


# loadng the embedding model
embeddings = HuggingFaceEmbeddings()

# Load PDF documents from the correct directory
loader = DirectoryLoader(path="data",
                         glob="*.pdf",
                         loader_cls=UnstructuredFileLoader)
documents = loader.load()

# Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=4000,
                                      chunk_overlap=500)
text_chunks = text_splitter.split_documents(documents)

# Store embeddings in the correct directory
vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory=vector_db_path
)

print("Documents Vectorized and Stored in:", vector_db_path)




