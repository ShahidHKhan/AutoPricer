import modal

from .base_agent import BaseAgent

MODAL_APP_NAME = "auto-pricer-service"
MODAL_CLASS_NAME = "AutoPricer"


class SpecialistAgent(BaseAgent):
    """Wraps the Modal-deployed fine-tuned Llama-3.2-3B model.

    Deliberately does NOT pass RAG context. Week 8 Day 3 testing (15-car
    sample) showed injecting comparable-price context into this model made
    predictions worse on average ($1,691.80 MAE without RAG vs $2,490.93
    with RAG) — the model was never trained on a prompt shape that includes
    comps, so it anchors on them crudely rather than reasoning about them.
    See week8_day3_decision.md for the full writeup. Revisit only after
    retraining with a comps-aware prompt format.
    """

    def __init__(self):
        AutoPricer = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
        self.pricer = AutoPricer()

    def price(self, description: str) -> float:
        return self.pricer.price.remote(description)
