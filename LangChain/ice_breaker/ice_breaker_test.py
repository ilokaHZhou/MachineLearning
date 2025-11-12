from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv()
    print("API_KEY:", os.getenv("OPENAI_API_KEY"))