from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredExcelLoader
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


# Directory containing the Excel files
directory = r"./dataset/"

# Initialize a list to store the documents from all Excel files
all_docs = []

# Iterate through all files in the directory
for filename in os.listdir(directory):
    # Check if the file is an Excel file (you can add more extensions if needed)
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        # Create the full file path
        file_path = os.path.join(directory, filename)
        # Load the Excel file using UnstructuredExcelLoader
        loader = UnstructuredExcelLoader(file_path, mode="elements")
        # Load the documents and add to the list
        docs = loader.load()
        all_docs.extend(docs)  #


# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_docs)

# Embed
vectorstore = FAISS.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings(api_key = os.getenv('OPENAI_API_KEY')))

vectorstore.save_local("faiss_index")