from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def upload_htmls():
    
    # Load all the HTML pages in the given folder
    loader = DirectoryLoader(path="hr-policies")
    documents = loader.load()
    print(f"{len(documents)} Pages Loaded")

    # Split loaded documents into Chunks using CharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )


    split_documents = text_splitter.split_documents(documents=documents)
    print(f"Split into {len(split_documents)} Documents...")

    print(split_documents[0].metadata)

    # Upload chunks as vector embeddings into FAISS
    embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
    # embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.from_documents(split_documents, embedding)
    # Save the FAISS DB locally
    db.save_local("faiss_index")

def faiss_query():
    embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
    #embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    new_db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

    query = "Explain the Candidate Onboarding process."
    docs = new_db.similarity_search(query)

    # Print all the extracted Vectors from the above Query
    for doc in docs:
        print("##----- Page ------##")
        print(doc.metadata['source'])
        print("##----- Content ------##")
        print(doc.page_content)

if __name__ == "__main__":
    upload_htmls()
    #faiss_query()

