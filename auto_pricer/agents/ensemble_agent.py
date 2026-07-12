from .base_agent import BaseAgent
from .frontier_agent import FrontierAgent, VECTORSTORE_PATH
from .specialist_agent import SpecialistAgent


class EnsembleAgent(BaseAgent):
    """Weighted average of SpecialistAgent (fine-tuned, no RAG) and
    FrontierAgent (Gemini + RAG).

    Week 8 Day 3 testing (15-car sample) showed this beats both individual
    agents: Specialist MAE $1,691.80, Frontier MAE $1,599.20, Ensemble
    (50/50) MAE $1,202.10. The two agents tend to miss in different
    directions on the same cars, so averaging reduces variance rather than
    just splitting the difference. See week8_day3_decision.md for details.

    specialist_weight defaults to 0.5 (the tested/validated split). Only
    change this if you've re-run the benchmark and have evidence a different
    weighting does better — don't guess at weights without data.

    vectorstore_path is forwarded to FrontierAgent — override for a cloud
    deployment (e.g. a mounted Modal Volume path), otherwise uses the same
    local default as before.
    """

    def __init__(self, specialist_weight: float = 0.5, vectorstore_path: str = VECTORSTORE_PATH):
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(vectorstore_path=vectorstore_path)
        self.specialist_weight = specialist_weight

    def price(self, description: str) -> float:
        p_specialist = self.specialist.price(description)
        p_frontier = self.frontier.price(description)
        return (
            self.specialist_weight * p_specialist
            + (1 - self.specialist_weight) * p_frontier
        )
