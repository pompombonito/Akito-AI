# Akito Research Assistant

Akito is a local Computer Engineering research assistant. It answers from files indexed in ChromaDB and can optionally search the web, save topic-related pages into its research library, and inspect images with a local vision model.

## Requirements

- Python 3.10 or newer
- Node.js 20.19 or newer and pnpm (or npm)
- Ollama installed and running
- Tesseract OCR installed for scanned PDFs (the current OCR setting in `rag.py` points to `C:\Program Files\Tesseract-OCR\tesseract.exe`)
- Internet access for `/search`, `/web`, and `/learn`

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

To enable local push-to-talk transcription, install its optional dependencies:

```powershell
python -m pip install -r requirements-speech.txt
```

The first transcription downloads Whisper Tiny into the local model cache; after that, audio is transcribed by the local API and is not sent to a speech service.

Create the assistant model and fetch a vision model:

```powershell
ollama create devon-rs-ai-assistant -f Modelfile
ollama pull qwen2.5vl:3b
```

The vision model can be changed with the `AKITO_VISION_MODEL` environment variable. Set `AKITO_MODEL` to change the regular chat model.

## Run

Install frontend packages once:

```powershell
pnpm --dir frontend install
```

Start the API and React app in two terminals:

```powershell
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

```powershell
pnpm --dir frontend dev
```

Open `http://127.0.0.1:5173`. For a single local server, build with `pnpm --dir frontend build`, then run Uvicorn; FastAPI will serve `frontend/dist`. The existing terminal workflow remains available with `python main.py`.

The React interface provides chat with visible, cancellable task stages; locally saved conversation history; recorded push-to-talk transcribed by local Whisper and optional spoken replies; web search when enabled or explicitly requested in chat; local file ingestion; image analysis; schematic review with web curation; an embedded prototype designer with labeled SVG/PNG and connectivity-graph export; a research notebook; a structured experiment log with CSV export; and user-controlled memory. Chat requests are routed through task-specific instructions for embedded design, programming/debugging, research methods, ideation, data analysis, and image analysis. These are focused workflows around the configured local model, not a separate hidden-reasoning engine.

Knowledge Library includes continuous public-web source discovery. When enabled, Akito rotates through broad embedded-research topics and use-case queries indefinitely, advances result offsets without wrapping back to earlier pages, and moves to a new domain after learning a page. There is no app-level site or session cap; the user toggle is the stop control. Already learned domains are skipped, while failed or rejected URLs receive only a temporary 24-hour URL-specific cooldown. The relevance check accepts related terms such as TinyML/on-device AI and requires technical context; dictionaries, lyrics, encyclopedias, and shopping/social sites are filtered out. Search covers indexed results exposed by public search providers, not every page on the Internet. The API process must remain running and the computer must be awake and online; closing the browser alone does not stop the worker. Web pages are downloaded into the local retrieval library for citation, not used to fine-tune model weights. Cleanup removes byte-identical duplicates only inside `documents/learned/` and Chroma entries whose source files no longer exist. User uploads, notes, experiments, memory, and chat history are preserved.

Put local `.pdf`, `.txt`, `.md`, and `.csv` files under `documents/`, then run `python vector_db.py` to index or refresh them. Files added in the UI go into `documents/user_uploads/`; downloaded web pages go into `documents/learned/`; reviewed schematics, notebook entries, chat history, and experiment measurements are also saved below `documents/`. Scanned PDF pages use Tesseract OCR. CSV rows are indexed with their column names for retrieval; when asked to analyze data, the API also computes bounded descriptive statistics directly from local CSVs (up to 50,000 rows per file) and experiment-log measurements. Search results preserve file and page references; learned web pages also retain their original URL.

## Commands

- `/search <query>` searches the web and lists result titles and links.
- `/web <question>` searches the web and asks Akito to answer using those results, with URLs for citations.
- `/learn <topic>` paginates through distinct search results until the provider returns no new results, saves every readable matching page under `documents/learned/`, and indexes them for future local questions. In the Web Trainer, use **Load more results** to keep paging; all loaded results are selectable, with no app-level page cap.
- `/learn-model [focus]` uses the expertise areas in `Modelfile` as the default learning topic, or narrows discovery to the supplied focus.
- `/image "path to image" [question]` asks the local vision model about an image. Image paths with spaces should be quoted.
- `/help` lists commands.

Web learning downloads page text for local search. It does not retrain or modify the language model weights. Review saved pages in `documents/learned/` like other research sources. Explicit chat wording such as “search the web for …” or “look this up online” turns on web search for that request even if the sidebar toggle is off. The app reports a failed search instead of presenting the answer as web-verified.

## Schematic and prototype review

The schematic reader uses the local vision model to draft components, visible pin labels, connections, and uncertainty markers. Review and correct the extraction before adding it to the local library. Cross-checking searches local references and, when Web Search is enabled, curates relevant web results from the extracted component labels. AKITO fetches bounded page text when available and labels snippet-only sources as preliminary. Select useful results to download readable page text into the local library. Web fetches reject non-public hosts and unsafe redirects.

Prototype drafts can use both local and web references and export a labeled conceptual block diagram as SVG or PNG, a component BOM, and a versioned component/connection graph. Static checks catch incomplete or mismatched endpoint labels; they do not perform electrical-rule checks. The graph is draft interchange data, not a verified electrical CAD schematic or manufacturing-ready netlist. Check pin mappings, voltage/current limits, and electrical rules in an EDA tool before building. Novelty is presented as a research hypothesis and must be checked against the literature.

Memory entries are created, edited, and deleted by the user in the Memory page and stored in `documents/memory.json`. Only a small set of notes that match the current question is added as context. Push-to-talk records only while the button is active, stops after 30 seconds, and adds a transcript to the composer for review; AKITO does not listen continuously or activate a wake word. Chat text and citations are automatically saved in `documents/conversations/`; reopen a conversation to continue with its recent turns, while image bytes remain unstored. The Experiment Log appends measurements to `documents/experiments.json`, displays the newest 100, and exports the full dataset as CSV. When you request numeric analysis, AKITO reports calculated descriptive statistics for exact numeric values in the selected records or CSV. It does not infer missing measurements or treat descriptive statistics as evidence of significance.

## Retrieval evaluation

Run `python scripts/evaluate_retrieval.py` to measure whether the expected source files appear in the top eight local search results for the cases in `evaluation/retrieval_cases.jsonl`. This is a source-recall check, not an evaluation of generated answer truth or electrical safety. Add reviewed questions and expected filenames as the research library grows.
