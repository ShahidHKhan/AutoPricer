import modal
from modal import Volume, Image

app = modal.App("auto-pricer-service")
image = Image.debian_slim().pip_install(
    "torch", "transformers", "bitsandbytes", "accelerate", "peft"
)
secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
HF_USER = "ShahidHKhan"
PROJECT_RUN_NAME = "autopricer-full"
REVISION = "158b05b"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"
CACHE_DIR = "/cache"

MIN_CONTAINERS = 0  # sleeps after ~2 min idle; flip to 1 later to keep always-warm

PREFIX = "Price is $"
QUESTION = "What does this used car cost to the nearest dollar?"

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets,
    gpu=GPU,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume},
)
class AutoPricer:
    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=quant_config, device_map="auto"
        )
        self.fine_tuned_model = PeftModel.from_pretrained(
            self.base_model, FINETUNED_MODEL, revision=REVISION
        )

    @modal.method()
    def price(self, description: str, context: str = "") -> float:
        import re
        import torch
        from transformers import set_seed

        set_seed(42)
        if context:
            prompt = f"{QUESTION}\n\n{description}\n\n{context}\n\n{PREFIX}"
        else:
            prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.fine_tuned_model.generate(inputs, max_new_tokens=5)
        result = self.tokenizer.decode(outputs[0])
        contents = result.split(PREFIX)[1].replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0


# --- Chrome extension endpoint: extraction + EnsembleAgent pricing ---

extraction_image = (
    Image.debian_slim()
    .pip_install(
        "torch", "sentence-transformers", "chromadb", "litellm", "pydantic",
        "fastapi[standard]", "datasets",
    )
    .add_local_python_source("auto_pricer")
)

vectorstore_volume = Volume.from_name("cars-vectorstore-vol")
VECTORSTORE_MOUNT = "/vol/cars_vectorstore"

extraction_secrets = [
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("gemini-secret"),  # <-- confirm this name matches `modal secret list`
]


@app.cls(
    image=extraction_image,
    secrets=extraction_secrets,
    timeout=120,
    volumes={"/vol": vectorstore_volume},
)
class PricingEndpoint:
    @modal.enter()
    def setup(self):
        # Runs once per container lifetime, not per-request — this is where
        # SentenceTransformer + Chroma client loading gets hoisted to.
        from auto_pricer.agents import EnsembleAgent
        self.ensemble = EnsembleAgent(vectorstore_path=VECTORSTORE_MOUNT)

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel

        web_app = FastAPI()
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["POST"],
            allow_headers=["*"],
        )

        class PriceRequest(BaseModel):
            text: str

        @web_app.post("/estimate")
        def estimate(req: PriceRequest):
            from auto_pricer.agents.extraction import (
                extract_car_details, to_car, format_summary_for_pricing,
            )

            extracted = extract_car_details(req.text)
            car = to_car(extracted)
            car.summary = format_summary_for_pricing(extracted)

            estimated_price = self.ensemble.price(car.summary)
            asking_price = extracted.price
            delta_pct = (
                (asking_price - estimated_price) / estimated_price * 100
                if estimated_price else 0.0
            )

            if delta_pct > 10:
                verdict = "overpriced"
            elif delta_pct < -10:
                verdict = "underpriced"
            else:
                verdict = "fair"

            return {
                "asking_price": asking_price,
                "estimated_price": round(estimated_price, 2),
                "verdict": verdict,
                "delta_pct": round(delta_pct, 1),
            }

        return web_app