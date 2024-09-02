from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


documents = SimpleDirectoryReader("data").load_data()

# Embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# LLM
Settings.llm = Ollama(model="llama3.1:8b", request_timeout=360.0)

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(streaming=True)
# streaming=True: Allows you to start printing or processing the beginning of the response before the full response is finished.

response = query_engine.query("What did the author do growing up?")
# print(response)
response.print_response_stream()
