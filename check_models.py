import google.generativeai as genai
import os
import logging
from dotenv import load_dotenv
load_dotenv()
os.environ['API_KEY'] = os.getenv('GEMINI_API_KEY')

API_KEY = os.environ.get('API_KEY', '')


if API_KEY == "":
    print("Error: Please open check_models.py and paste your actual API key into the script.")
else:
    try:
        genai.configure(api_key=API_KEY)
        
        print("--- Listing available models for your API key ---")
        
        # List all models
        found_models = False
        for m in genai.list_models():
            # Check if the model supports the 'generateContent' method
            if 'generateContent' in m.supported_generation_methods:
                print(f"✅ Found usable model: {m.name}")
                found_models = True
                
        print("--------------------------------------------------")
        
        if found_models:
            print("\nSUCCESS: Your key can see models.")
            print("Make sure one of the model names above (e.g., 'models/gemini-1.0-pro')")
            print("is the one you are using in 'question_generator.py'.")
        else:
            print("\nFAILURE: Your key is valid, but no models were found.")
            print("This confirms your Google Cloud project's Billing/API setup is the issue.")

    except Exception as e:
        logging.error(f"An error occurred. This often means the API key is invalid or the project setup is wrong.")
        logging.error(f"Error details: {e}")