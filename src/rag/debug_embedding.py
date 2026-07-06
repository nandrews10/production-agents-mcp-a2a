from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

print("Loading embedding model...")

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

print("Model loaded.")

query = "incident response"

print("Creating embedding...")

embedding = embedding_function([query])

print("Embedding created.")
print(type(embedding))
print(len(embedding))
print(len(embedding[0]))
print(embedding[0][:5])