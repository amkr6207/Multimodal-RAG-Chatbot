import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

def test_connections():
    print("--- Testing Connections ---")
    
    # 1. Test Google Gemini API
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_key or gemini_key == "your_google_gemini_api_key_here":
        print("🟡 Google API Key not found (skipping Gemini)")
    else:
        print("Checking Google Gemini API...")
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents="Say hello!"
            )
            print(f"✅ Google Gemini connection successful! Response: {response.text.strip()}")
        except Exception as e:
            print(f"❌ Google Gemini connection failed: {e}")

    # 2. Test Groq API
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("🟡 Groq API Key not found in .env (skipping Groq)")
    else:
        print("Checking Groq AI API...")
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "Say hello!"}]
            )
            print(f"✅ Groq connection successful! Response: {completion.choices[0].message.content.strip()}")
        except Exception as e:
            print(f"❌ Groq connection failed: {e}")

    # 3. Test MongoDB Connection
    mongodb_uri = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
    if not mongodb_uri or "mongodb+srv" not in mongodb_uri:
        print("❌ MongoDB Atlas URI not found in .env")
    else:
        try:
            client = MongoClient(mongodb_uri)
            # Send a ping to confirm a successful connection
            client.admin.command('ping')
            print("✅ MongoDB Atlas connection successful!")
        except Exception as e:
            print(f"❌ MongoDB Atlas connection failed: {e}")

if __name__ == "__main__":
    test_connections()
