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
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        self.vector_search = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
        )
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def get_context(self, query, doc_ids=None):
        """Retrieve relevant document chunks from MongoDB"""
        if doc_ids:
            if isinstance(doc_ids, str):
                doc_ids = [doc_ids]
            doc_ids = [doc_id for doc_id in doc_ids if doc_id]
        else:
            doc_ids = None

        if doc_ids:
            pre_filter = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}
            try:
                results = self.vector_search.similarity_search(
                    query,
                    k=5,
                    pre_filter=pre_filter
                )
            except Exception:
                # Fallback when Atlas vector index does not support this pre_filter path yet.
                candidates = self.vector_search.similarity_search(query, k=75)
                selected = set(doc_ids)
                results = [doc for doc in candidates if doc.metadata.get("doc_id") in selected][:5]
        else:
            results = self.vector_search.similarity_search(query, k=5)
        context = "\n\n".join([doc.page_content for doc in results])
        return context

    def generate_answer(self, query, doc_ids=None):
        """Generate answer using Groq with retrieved context"""
        context = self.get_context(query, doc_ids=doc_ids)
        if not context:
            return "I couldn't find relevant context in the selected document(s). Please re-ingest and try again."
        
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
