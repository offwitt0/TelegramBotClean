import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the SOP content
with open("sop_cleaned.txt", "r", encoding="utf-8") as f:
    sop_text = f.read()

doc = Document(page_content=sop_text, metadata={"source": "guest_sop"})

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents([doc])

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("guest_kb_vectorstore")
print("✅ Vectorstore saved.")
