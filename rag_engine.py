import os
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Configuration
DB_NAME = os.getenv("DB_NAME", "rag_chatbot")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_embeddings")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME", "vector_index")
MONGODB_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")

class RAGEngine:
    def __init__(self):
        self.client = MongoClient(MONGODB_URI)
        self.collection = self.client[DB_NAME][COLLECTION_NAME]
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_search = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
        )
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def get_context(self, query):
        """Retrieve relevant document chunks from MongoDB"""
        results = self.vector_search.similarity_search(query, k=5)
        context = "\n\n".join([doc.page_content for doc in results])
        return context

    def generate_answer(self, query):
        """Generate answer using Groq with retrieved context"""
        context = self.get_context(query)
        
        prompt = f"""
        You are a helpful AI Assistant. Use the provided context to answer the user's question.
        If the context doesn't contain the answer, say "I don't have enough information in the documents to answer that."
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
        
        completion = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        return completion.choices[0].message.content

if __name__ == "__main__":
    engine = RAGEngine()
    # print(engine.generate_answer("What is the main topic of the document?"))
    print("RAG Engine ready.")
