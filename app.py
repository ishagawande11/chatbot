from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.documents import Document
import os

# Load text files
DATA_DIR = "data"
documents = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".txt"):
        with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
            documents.append(Document(page_content=f.read(), metadata={"source": filename}))

print(f"Loaded {len(documents)} documents.")

# Split documents for better retrieval accuracy
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

# Embeddings model
embedding = OllamaEmbeddings(model="mxbai-embed-large")

# Create / update vector database
vectordb = Chroma.from_documents(split_docs, embedding)

# LLM model
llm = OllamaLLM(model="llama3")

# Retrieval QA with strict mode (NO hallucination)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

print("Ask a question. Type 'exit' to quit.\n")

# User loop
while True:
    query = input("You: ")

    if query.lower() in ["exit", "quit", "bye"]:
        print("\nChatbot: Goodbye 👋")
        break

    result = qa.invoke({"query": query})
    answer = result.get("result", "").strip()

    if answer:
        print("\nChatbot:", answer)
    else:
        print("\nChatbot: I couldn't find that information in the provided documents.")

    