"""
Test script to verify your Gemini API key is working
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()
# Method 1: Set API key directly in code (for testing)
# Method 2: Or set it via environment variable before running:
# export GOOGLE_API_KEY="your-key-here"

def test_gemini_connection():
    """Test if Gemini API is working"""
    print("🔑 Testing Gemini API Key...")
    print(f"API Key (first 10 chars): {os.environ.get('GOOGLE_API_KEY', 'NOT SET')[:10]}...")
    
    try:
        # Initialize the model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=os.environ.get("GOOGLE_API_KEY")  # Explicitly pass the key
        )
        
        print("✅ Model initialized successfully!")
        
        # Test a simple query
        print("\n🧪 Testing with a simple question...")
        response = llm.invoke("Say 'Hello, the API key is working!' in exactly those words.")
        
        print(f"\n✅ SUCCESS! Response received:")
        print(f"📝 {response.content}")
        print("\n🎉 Your Gemini API key is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n🔍 Troubleshooting:")
        print("1. Check your API key at: https://aistudio.google.com/app/apikey")
        print("2. Make sure you've enabled the Gemini API")
        print("3. Verify the key is correctly copied (no extra spaces)")
        print("4. Try creating a new API key if this one doesn't work")
        return False

if __name__ == "__main__":
    test_gemini_connection()