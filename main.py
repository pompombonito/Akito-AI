"""Akito: Computer Engineering AI assistant"""

import os
from pathlib import Path
import shlex

import chromadb
from sentence_transformers import SentenceTransformer

from assistant_core import add_web_knowledge, answer_stream, classify_research_task, format_web_context, requests_web_search
from internet_search import search_web


MODEL = os.getenv("AKITO_MODEL", "devon-rs-ai-assistant")
VISION_MODEL = os.getenv("AKITO_VISION_MODEL", "qwen2.5vl:3b")
DATABASE_FOLDER = "database"
COLLECTION_NAME = "research_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NUMBER_OF_RESULTS = 5


def load_model_profile():
    """Read the system instructions from the project's Ollama Modelfile."""
    modelfile = Path("Modelfile")
    if not modelfile.exists():
        return "You are a careful Computer Engineering research assistant."
    contents = modelfile.read_text(encoding="utf-8")
    marker = 'SYSTEM """'
    start = contents.find(marker)
    if start < 0:
        return contents
    start += len(marker)
    end = contents.find('"""', start)
    return contents[start:end].strip() if end >= 0 else contents[start:].strip()


PROFILE = load_model_profile()

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=DATABASE_FOLDER)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
print(f"Research database ready: {collection.count()} chunks")


def search_documents(query, number_of_results=NUMBER_OF_RESULTS):
    if collection.count() == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    query_embedding = embedding_model.encode([query]).tolist()
    return collection.query(query_embeddings=query_embedding, n_results=min(number_of_results, collection.count()))


def build_context(results):
    sources = []
    for document, metadata in zip(results["documents"][0], results["metadatas"][0]):
        citation = f"{metadata.get('filename', 'local file')}, p. {metadata.get('page', 1)}"
        if metadata.get("source_url"):
            citation += f" ({metadata['source_url']})"
        sources.append(f"Source: {citation}\nContent:\n{document}")
    return "\n\n".join(sources) or "No matching local files were found."


def ask_akito(question, context, model=MODEL, images=None):
    print("\nAkito: ", end="", flush=True)
    for fragment in answer_stream(
        question,
        context,
        model=model,
        images=images,
        task_type=classify_research_task(question, bool(images)),
    ):
        print(fragment, end="", flush=True)
    print("\n")


def web_context(results):
    return "\n\n".join(
        f"Title: {item['title']}\nURL: {item['url']}\nSummary: {item['snippet']}"
        for item in results
    ) or "No search results were found."


def show_search_results(results):
    if not results:
        print("No search results found.")
        return
    for index, item in enumerate(results, start=1):
        print(f"{index}. {item['title']}\n   {item['url']}\n   {item['snippet']}\n")


def learn_from_web(topic):
    """Find a few topic-relevant pages, save their text, and add them to RAG."""
    saved, skipped = add_web_knowledge(topic, max_pages=3)
    if saved:
        from vector_db import index_documents
        count = index_documents()
        print(f"Added {len(saved)} relevant web page(s); the local library now has {count} indexed chunks.")
    else:
        print(skipped or "No relevant readable pages could be saved.")


def model_topics():
    """Collect the profile's declared expertise topics from Modelfile."""
    lines = PROFILE.splitlines()
    in_expertise = False
    topics = []
    for line in lines:
        normalized = line.strip()
        if normalized.lower().startswith("primary areas of expertise"):
            in_expertise = True
            continue
        if in_expertise and normalized.startswith("-"):
            topics.append(normalized.lstrip("- "))
        elif in_expertise and topics and normalized:
            break
    return topics


def image_question(arguments):
    parts = shlex.split(arguments, posix=False)
    if not parts:
        print('Usage: /image "path to image" [question]')
        return
    image_path = parts[0].strip('"')
    question = " ".join(parts[1:]) or "Describe this image and identify its important contents."
    path = Path(image_path).expanduser()
    if not path.is_file():
        print(f"Image file not found: {path}")
        return
    ask_akito(question, "Answer using the image provided. Describe visible details and state uncertainty when appropriate.", model=VISION_MODEL, images=[str(path.resolve())])


def print_help():
    print("""Commands:
  /search <query>        Search the internet and list source links
  /web <question>        Search the internet, then answer with linked sources
  /learn <topic>         Download up to 3 relevant pages and index them locally
  /learn-model [focus]   Learn using Modelfile expertise areas (optionally focused)
  /image <path> [question]  Ask the vision model about an image
  /help                  Show this help
  exit                   Quit

Ordinary questions use the local indexed files. Run `python vector_db.py` after
adding local PDFs, text files, Markdown files, or CSV datasets under documents/.
""")


print("\nDevon's AI Research Assistant 'Akito'")
print("Type /help for commands, or 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "exit":
        print("Goodbye.")
        break

    try:
        command, _, arguments = user_input.partition(" ")
        if command == "/help":
            print_help()
        elif command == "/search":
            show_search_results(search_web(arguments))
        elif command == "/web":
            results = search_web(arguments)
            ask_akito(arguments, web_context(results))
        elif command == "/learn":
            if not arguments.strip():
                print("Usage: /learn <topic>")
            else:
                learn_from_web(arguments.strip())
        elif command == "/learn-model":
            topics = model_topics()
            focus = arguments.strip() or ", ".join(topics)
            if not focus:
                print("No expertise topics found in Modelfile; use /learn <topic>.")
            else:
                learn_from_web(focus)
        elif command == "/image":
            image_question(arguments)
        else:
            context = build_context(search_documents(user_input))
            if requests_web_search(user_input):
                web_results = search_web(user_input)
                context += "\n\nWEB SEARCH RESULTS:\n" + format_web_context(web_results)
            ask_akito(user_input, context)
    except Exception as exc:
        print(f"\nAn error occurred: {exc}\n")
