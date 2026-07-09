# AutoPricer — Week 8 Extension Instructions

This file documents all concepts from Week 8 of the LLM Engineering course and provides step-by-step implementation instructions for integrating them into the AutoPricer project. When working in the Claude project, also reference `autopricer_instructions_v2.md` (the core Days 1-7 plan) and the Week 8 `.ipynb` files uploaded alongside this document.

---

## All Week 8 Concepts (Reference)

### Category 1 — Cloud Deployment with Modal.com
- Modal basics: `@app.function`, running code remotely on cloud hardware
- Ephemeral apps (run once, spin down) vs Deployed apps (persistent, callable via API)
- Deploying GPU functions (`GPU="T4"`) for inference without a local GPU
- Modal Secrets management (storing HF_TOKEN, API keys securely in Modal)
- Modal Volumes: persistent storage for model weights so the model doesn't re-download every cold start
- `MIN_CONTAINERS` — keeping a container warm (always-on vs cold-start tradeoff)
- Warm-up control: `update_autoscaler(scaledown_window=...)` for managing when containers sleep
- Modal Class pattern (`@app.cls` + `@modal.enter()`) — loading model once at startup, reusing across requests
- Calling Modal functions from local Python: `modal.Function.from_name()`, `modal.Cls.from_name()`
- Region selection (`region="eu"`) for geographic control of where code runs

### Category 2 — RAG (Retrieval Augmented Generation)
- Semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2` (free, local, 384-dimensional vectors)
- ChromaDB as a persistent local vector database (`chromadb.PersistentClient`)
- Populating a vector store in batches from your training dataset
- Similarity search at inference time: finding the N most similar items to a query
- Building a context string from retrieved similar items to inject into the LLM prompt
- RAG as an "inference-time" technique vs fine-tuning as a "training-time" technique (complementary, not mutually exclusive)
- t-SNE dimensionality reduction for visualizing high-dimensional vector data in 2D/3D (Plotly scatter plots)

### Category 3 — Agentic Architecture / Multi-Agent Systems
- Agent as a class with a consistent interface (e.g. `.price()` method)
- SpecialistAgent: wraps the Modal-deployed fine-tuned LLM
- FrontierAgent: wraps a frontier LLM (Gemini/GPT) with RAG context injection
- NeuralNetworkAgent: wraps the local deep neural network
- EnsembleAgent: combines multiple agents with weighted averaging
- ScannerAgent: scrapes external sources for real-world data
- MessagingAgent: sends push notifications
- AutonomousPlanningAgent: LLM with tool-calling to orchestrate multi-step tasks
- Framework orchestrator: top-level class managing agents, memory, ChromaDB

### Category 4 — Tool Calling / Function Calling (Agentic Loops)
- Defining tools as JSON schema (`name`, `description`, `parameters`)
- The agentic loop: LLM decides to call a tool → you execute it → feed result back → repeat
- `handle_tool_call()` dispatcher pattern
- Structured outputs with Pydantic models for reliable JSON extraction
- Planning prompts: instructing the LLM to reason step-by-step autonomously

### Category 5 — Push Notifications (Pushover)
- Pushover API: real-time push notifications to your phone from Python
- `PUSHOVER_USER` and `PUSHOVER_TOKEN` in `.env`
- Simple `requests.post()` pattern to fire a notification
- Use case: alerting the user when a prediction completes, or a deal/anomaly is flagged

### Category 6 — Gradio UI
- Building a real-time interactive dashboard with `gr.Blocks`
- `gr.Dataframe` for displaying results tables
- `gr.State` for persisting in-memory state between Gradio events
- `gr.Timer` for polling/auto-refreshing on a schedule
- Threading + queues for running long background tasks without blocking the UI
- Live log streaming into a Gradio HTML component
- `gr.Plot` for embedding Plotly charts inside the UI
- Row selection events triggering actions
- Custom HTML log rendering with colored output

### Category 7 — Memory / Persistence
- JSON-file-based agent memory: persisting results across runs
- Reading/writing memory on startup/shutdown
- `reset_memory()` utility for testing

### Category 8 — RSS Feed / Web Scraping
- Scraping structured data from external sources
- `ScrapedDeal.fetch()` as a pattern for a data-collection agent
- Prompt engineering for extracting structured price/description from messy scraped text
- Filtering for high-confidence data (e.g. excluding ambiguous price descriptions)

---

## Implementation Plan

---

## WEEK 8, DAY 1 — Cloud Deployment with Modal.com

### Goal
Deploy your fine-tuned Llama-3.2-3B model (trained in Week 7) as a persistent GPU-backed service on Modal, so it can be called from anywhere — locally, from other agents, or eventually from a UI — without needing your local RTX 2080.

### Reference files
`pricer_ephemeral.py`, `pricer_service.py`, `pricer_service2.py`, `hello.py`, `llama.py`, `day1.ipynb`

### Why this matters for AutoPricer
Your fine-tuned model currently lives in a Colab notebook or on HF Hub. To use it in a real system (especially as one component in an ensemble or agent pipeline later), you need it deployed as a callable service. Modal gives you a T4 GPU on demand with your model loaded and ready to respond in seconds, without running a GPU 24/7.

### Step-by-step

**Step 1 — Set up Modal**
- Sign up at modal.com
- In your terminal: `pip install modal`
- Run: `modal token new` (follow the prompts, or manually add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` to your `.env`)
- Add your HF token as a Modal Secret:
  - Go to modal.com → Secrets → New Secret
  - Name it `huggingface-secret`, Key: `HF_TOKEN`, Value: your HF token

**Step 2 — Test Modal is working (hello.py pattern)**
```python
import modal
from modal import Image

app = modal.App("hello")
image = Image.debian_slim().pip_install("requests")

@app.function(image=image)
def hello() -> str:
    import requests
    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    return f"Hello from {data['city']}, {data['country']}!"
```
Run with: `with app.run(): result = hello.remote()`

**Step 3 — Create `auto_pricer_service.py` (ephemeral version first)**

Adapt `pricer_ephemeral.py` for your AutoPricer model:
```python
import modal
from modal import Image

app = modal.App("auto-pricer")
image = Image.debian_slim().pip_install(
    "torch", "transformers", "bitsandbytes", "accelerate", "peft"
)
secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
HF_USER = "ShahidHKhan"
PROJECT_RUN_NAME = "autopricer-<your-run-name>"  # from your Week 7 fine-tune
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"

PREFIX = "Price is $"
QUESTION = "What does this used car cost to the nearest dollar?"

@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def price(description: str) -> float:
    import re, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel

    prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant_config, device_map="auto"
    )
    fine_tuned = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)

    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = fine_tuned.generate(inputs, max_new_tokens=5)
    result = tokenizer.decode(outputs[0])
    contents = result.split(PREFIX)[1].replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
    return float(match.group()) if match else 0
```
Test it:
```python
with app.run():
    result = price.remote("2017 Jeep Compass SUV, 17933 miles, gasoline, automatic, no accidents")
    print(result)
```

**Step 4 — Upgrade to persistent service with model caching (`pricer_service2.py` pattern)**

This is the production version — model loads once at container startup via `@modal.enter()`, then all subsequent calls reuse it. Much faster after the first cold start:
```python
from modal import Volume, Image

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)
CACHE_DIR = "/cache"
MIN_CONTAINERS = 0  # set to 1 to keep always warm (uses credits)

@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets, gpu=GPU, timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume}
)
class AutoPricer:
    @modal.enter()
    def setup(self):
        # loads model once when container starts
        ...

    @modal.method()
    def price(self, description: str) -> float:
        # fast inference using already-loaded model
        ...
```
Deploy: `modal deploy auto_pricer_service.py`

Call from anywhere:
```python
AutoPricer = modal.Cls.from_name("auto-pricer", "AutoPricer")
pricer = AutoPricer()
result = pricer.price.remote("2017 Jeep Compass...")
```

**Step 5 — Checklist**
- [ ] Modal account set up, token working
- [ ] `huggingface-secret` created in Modal with correct HF_TOKEN
- [ ] Ephemeral test passes (returns a real dollar amount)
- [ ] Persistent service deployed (`modal deploy`)
- [ ] Cold start tested (first call may take 1-10 mins to build image)
- [ ] Warm call tested (subsequent calls should be ~2-5 seconds)
- [ ] `MIN_CONTAINERS` decision made (0 = sleeps after 2 mins, 1 = always warm but costs credits)

---

## WEEK 8, DAY 2 — RAG (Retrieval Augmented Generation)

### Goal
Build a vector store from your training car listings, so that at inference time you can find the 5 most similar cars and inject them as context ("comparables") into the LLM prompt — improving price prediction accuracy beyond what fine-tuning alone achieves.

### Reference files
`day2.ipynb` (Week 8), `autopricer_instructions_v2.md` (Day 4 section on frontier models)

### Why RAG improves your model
Your fine-tuned Llama learned general patterns (SUVs cost more, high mileage reduces price, etc.) but works purely from memory. RAG gives the model real, specific comparable listings at prediction time — like a human appraiser using comps. The model can then reason: "this 2017 Compass with 18k miles is similar to these 5 listings that sold for $22k-$26k, so I estimate $24k." For cars specifically, comparables are even more powerful than for generic products because car prices are so sensitive to make/model/year/mileage combinations that vector similarity maps extremely well to price similarity.

### Prerequisite
You need your processed dataset with populated `summary` fields on HF Hub (the Gemini-rewritten descriptions from Day 2). These are what you'll embed — not the raw `full` blobs. Confirm Day 2 preprocessing is complete before starting this section.

### New notebook: `5_rag.ipynb`

**Step 1 — Install and import**
```
pip install chromadb sentence-transformers
```
```python
import os
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from litellm import completion
from dotenv import load_dotenv
import sys
sys.path.append("..")
from auto_pricer.car import Car

load_dotenv(override=True)

username = "ShahidHKhan"
train, val, test = Car.from_hub(f"{username}/cars_full_processed")
print(f"Loaded {len(train):,} training cars")
```

**Step 2 — Initialize the embedding model**
```python
# all-MiniLM: free, local, fast, 384-dimensional vectors
# Data never leaves your machine — good for personal projects
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Test it
vector = encoder.encode(["2017 Jeep Compass SUV, 18k miles, gasoline, automatic"])[0]
print(f"Vector shape: {vector.shape}")  # should be (384,)
```

**Step 3 — Create and populate the ChromaDB vector store**
```python
DB = "cars_vectorstore"
client = chromadb.PersistentClient(path=DB)

collection_name = "cars"
existing = [c.name for c in client.list_collections()]

if collection_name not in existing:
    collection = client.create_collection(collection_name)
    BATCH_SIZE = 1000
    from tqdm import tqdm

    for i in tqdm(range(0, len(train), BATCH_SIZE)):
        batch = train[i:i + BATCH_SIZE]
        documents = [car.summary for car in batch]
        vectors = encoder.encode(documents).astype(float).tolist()
        metadatas = [{"make": car.make, "year": car.year, "price": car.price} for car in batch]
        ids = [f"car_{i + j}" for j in range(len(batch))]
        collection.add(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)
    print(f"Populated vectorstore with {len(train):,} cars")

collection = client.get_or_create_collection(collection_name)
print(f"Collection has {collection.count():,} entries")
```
Note: this will take 15-30+ minutes on your 2080 for a full 270k-car dataset. The result is saved to disk permanently — you only need to do this once.

**Step 4 — Visualize the vector space with t-SNE (optional but great for understanding)**
```python
MAXIMUM_DATAPOINTS = 5_000  # keep low or it may crash — increase cautiously

result = collection.get(
    include=['embeddings', 'documents', 'metadatas'],
    limit=MAXIMUM_DATAPOINTS
)
vectors = np.array(result['embeddings'])
documents = result['documents']
makes = [m['make'] for m in result['metadatas']]

# t-SNE: reduces 384-dimensional vectors to 3D for visualization
tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
reduced = tsne.fit_transform(vectors)

# Color by make (top 8 most common)
from collections import Counter
top_makes = [m for m, _ in Counter(makes).most_common(8)]
COLORS = ['red','blue','green','orange','purple','cyan','brown','yellow']
colors = [COLORS[top_makes.index(m)] if m in top_makes else 'gray' for m in makes]

fig = go.Figure(data=[go.Scatter3d(
    x=reduced[:,0], y=reduced[:,1], z=reduced[:,2],
    mode='markers',
    marker=dict(size=2, color=colors, opacity=0.7),
    text=[f"Make: {m}<br>{d[:60]}..." for m, d in zip(makes, documents)],
    hoverinfo='text'
)])
fig.update_layout(title="Car Listings Vector Space (colored by make)", height=600)
fig.show()
```
What you should see: similar cars clustering together in 3D space — luxury brands in one region, trucks in another, budget sedans in another. This confirms your embeddings are capturing real semantic similarity.

**Step 5 — Similarity search function**
```python
def find_similars(car, n_results=5):
    vector = encoder.encode([car.summary]).astype(float).tolist()
    results = collection.query(
        query_embeddings=vector,
        n_results=n_results,
        include=['documents', 'metadatas']
    )
    documents = results['documents'][0]
    prices = [m['price'] for m in results['metadatas'][0]]
    return documents, prices
```

**Step 6 — Build the context string for RAG injection**
```python
def make_context(similar_docs, prices):
    context = "Here are some similar used car listings with their prices:\n\n"
    for doc, price in zip(similar_docs, prices):
        context += f"Price: ${price:,.0f}\n{doc}\n\n"
    return context.strip()

# Test it
docs, prices = find_similars(test[0])
print(make_context(docs, prices))
```

**Step 7 — RAG-augmented Gemini inference function**
```python
def messages_for_rag(car):
    docs, prices = find_similars(car)
    context = make_context(docs, prices)
    message = f"""Estimate the price of this used car. Respond with the price only, no explanation.

Car to price:
{car.summary}

{context}"""
    return [{"role": "user", "content": message}]

def gemini_rag(car):
    response = completion(
        model="gemini/gemini-2.5-flash",
        messages=messages_for_rag(car)
    )
    return response.choices[0].message.content
```

**Step 8 — Evaluate and compare**
```python
from auto_pricer.evaluator import evaluate  # your Day 3 evaluator

# RAG-augmented frontier model
evaluate(gemini_rag, test)

# Compare against zero-shot (no RAG) from Day 4
# Run both and record the dollar error difference
```

**Step 9 — Checklist**
- [ ] ChromaDB populated with full training set (one-time, saved to disk)
- [ ] t-SNE visualization confirms semantic clustering is working
- [ ] `find_similars()` returns genuinely similar cars (spot-check 10 examples manually)
- [ ] RAG prompt produces better results than zero-shot — record both dollar errors
- [ ] Add RAG results to your final `results.ipynb` comparison bar chart

---

## WEEK 8, DAY 3 — Agentic Architecture (Open-Ended / Expandable)

### Goal
Build a multi-agent system around your AutoPricer model, turning it from a notebook experiment into something that can take real-world inputs and produce real-world outputs autonomously. The specific features below are a starting framework — expand once Days 1 and 2 are complete and you know what you want to build.

### Reference files
`day3.ipynb`, `day4.ipynb` (Week 8), `deal_agent_framework.py`

### Core agent pattern (consistent interface for all agents)
Every agent should expose a `.price(description: str) -> float` method, making them interchangeable and composable:
```python
class BaseAgent:
    def price(self, description: str) -> float:
        raise NotImplementedError
```

### Confirmed feature idea: Marketplace Listing Scraper Agent
Take a URL from Facebook Marketplace, Craigslist, or Cars.com, scrape the page, extract structured car data, and feed it into the pricing pipeline.

High-level design:
```
URL input
    ↓
ScraperAgent.fetch(url)  →  raw HTML
    ↓
LLM tool call: extract make/model/year/mileage/condition/description
    ↓
Car object (matching your existing schema)
    ↓
EnsembleAgent.price(car.summary)  →  estimated price
    ↓
return price + "is this a good deal?" comparison
```

Key libraries to investigate: `requests`+`BeautifulSoup` for static pages, `playwright` or `selenium` for JavaScript-rendered pages (Facebook Marketplace will need this).

**Agent classes to build (suggested order):**

1. **SpecialistAgent** — wraps your Modal-deployed fine-tuned Llama (from Day 1)
2. **FrontierAgent** — wraps Gemini with RAG injection (from Day 2)
3. **ScraperAgent** — takes a URL, returns a structured `Car` object
4. **EnsembleAgent** — combines SpecialistAgent + FrontierAgent + your DeepNN with weighted averaging
5. **AutonomousPlanningAgent** (optional) — LLM orchestrator that uses tool-calling to decide which agents to invoke

### Tool calling pattern for the ScraperAgent
```python
scrape_tool = {
    "name": "scrape_listing",
    "description": "Scrapes a car listing URL and returns structured data",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL of the car listing"},
        },
        "required": ["url"]
    }
}

extract_tool = {
    "name": "extract_car_details",
    "description": "Given raw listing text, extracts make, model, year, mileage, condition",
    "parameters": { ... }
}
```

### Agentic loop skeleton
```python
messages = [system_message, user_message]
while True:
    response = llm.chat(messages=messages, tools=tools)
    if response.finish_reason == "tool_calls":
        results = handle_tool_call(response.message)
        messages.append(response.message)
        messages.extend(results)
    else:
        break
```

### Leave open for now
The exact feature set for Day 3+ should be decided after Day 1 (Modal) and Day 2 (RAG) are complete and working. Other possible agent features to consider once you're ready:
- Price history lookups (web search agent)
- "Is this a good deal?" comparison vs similar listings in the vector store
- Batch pricing multiple listings from a CSV or search result page
- Confidence scoring: how certain is the model? (variance across ensemble members)

---

## WEEK 8, OPTIONAL CATEGORIES — Reference for Later

### Category 5 — Push Notifications (Pushover)
**When to use:** if you build any async feature (a background agent that monitors a marketplace, or a batch job that takes hours) and you want to be notified on your phone when it finishes or finds something interesting.

**Quick setup:**
1. Sign up at pushover.net, create an Application/API Token
2. Add to `.env`: `PUSHOVER_USER=u...` and `PUSHOVER_TOKEN=a...`
3. Install on your phone via the Pushover app
4. Send from Python:
```python
import requests, os
def push(message):
    requests.post("https://api.pushover.net/1/messages.json", data={
        "user": os.getenv("PUSHOVER_USER"),
        "token": os.getenv("PUSHOVER_TOKEN"),
        "message": message
    })
push("AutoPricer: 2017 Jeep Compass estimated at $24,200")
```
**AutoPricer use cases:** notify when a long Gemini preprocessing run finishes; alert when a scraped listing looks underpriced vs your model's estimate.

### Category 6 — Gradio UI
**When to use:** when you want a shareable, interactive web interface for your project instead of just a notebook. Great for a portfolio demo.

**What to build for AutoPricer:**
- Text input box: paste a car description or URL
- Button: "Estimate Price"
- Output: predicted price from the ensemble
- Optional: dataframe showing the 5 RAG comparables used
- Optional: Plotly 3D t-SNE chart of the vector space embedded directly in the UI

**Quick skeleton:**
```python
import gradio as gr

def predict(description):
    # call your ensemble agent
    return f"Estimated price: ${price:,.0f}"

with gr.Blocks(title="AutoPricer") as ui:
    gr.Markdown("## AutoPricer — Used Car Price Estimator")
    description = gr.Textbox(label="Car Description or URL")
    output = gr.Textbox(label="Estimated Price")
    btn = gr.Button("Estimate")
    btn.click(predict, inputs=[description], outputs=[output])

ui.launch(inbrowser=True)
```
Reference `price_is_right.py` and `day5.ipynb` (Week 8) for the full threading/logging/live-update pattern when the prediction takes more than a second.

### Category 7 — Memory / Persistence
**When to use:** if you build any agent that runs repeatedly (e.g. a scanner that checks for new listings every hour) and needs to remember what it already found.

**Pattern:**
```python
import json, os

MEMORY_FILE = "autopricer_memory.json"

def read_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return []

def write_memory(data):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)
```
Reference `deal_agent_framework.py` for the full pattern including `reset_memory()`.

### Category 8 — RSS / Web Scraping Data Sources
**When to use:** if you want your AutoPricer to proactively find cars to price rather than waiting for user input. Could scrape Cars.com search results, Craigslist RSS feeds, or Facebook Marketplace (requires headless browser).

**Pattern from course:**
```python
import feedparser

def fetch_listings(rss_url):
    feed = feedparser.parse(rss_url)
    return [entry for entry in feed.entries]
```
Reference `day3.ipynb` (Week 8) for the full `ScrapedDeal.fetch()` pattern and the LLM-based structured data extraction prompt.

---

## Current Project Status (as of end of Week 7 / start of Week 8)

- Day 1 (Curation): ✅ `cars_raw` pushed to HF Hub (~107k cars)
- Day 2 (Preprocessing): ⚠️ Gemini rewriting was interrupted mid-run due to 503 outages — needs to be resumed and completed before RAG can be built
- Day 3 (Baselines/Traditional ML): 🔲 Not yet started
- Day 4 (Deep Learning + Frontier): 🔲 Not yet started
- Day 5 (Prompt datasets): 🔲 Not yet started
- Day 6-7 (QLoRA Fine-tuning on Colab): 🔲 Not yet started
- Week 8 Day 1 (Modal deployment): 🔲 Blocked on fine-tune completing first
- Week 8 Day 2 (RAG): 🔲 Blocked on Day 2 preprocessing completing first

**Next immediate action:** Resume and complete the Gemini preprocessing run (Day 2) by first running the 5-call latency diagnostic to confirm Gemini's endpoint is healthy, then kicking off the full batch with `max_workers=8-10` and the retry/failure-tracking code from the conversation history.
