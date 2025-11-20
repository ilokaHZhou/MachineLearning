import os
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_host = os.getenv("PINECONE_HOST")
pc = Pinecone(api_key=pinecone_api_key)

index = pc.Index(host=pinecone_host)

print(index.describe_index_stats())