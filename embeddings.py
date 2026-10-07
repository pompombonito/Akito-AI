from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


texts = [
    "ESP32 is a 2.4 GHz Wi-Fi and Bluetooth combo chip.",
    "The ESP32 contains Xtensa 32-bit LX6 microprocessors.",
    "The ESP32 includes Wi-Fi and Bluetooth connectivity."
]


print()
print("Creating embeddings...")

embeddings = model.encode(texts)

print("Embeddings created.")

print()
print("Number of texts:", len(embeddings))
print("Embedding dimensions:", len(embeddings[0]))