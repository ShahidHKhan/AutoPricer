from pydantic import BaseModel
from typing import Optional, Self
from datasets import Dataset, DatasetDict, load_dataset

PREFIX = "Price is $"
QUESTION = "What does this used car cost to the nearest dollar?"


class Car(BaseModel):
    title: str
    make: str
    model: str
    trim: Optional[str] = None
    year: int
    mileage: float
    price: float
    horsepower: Optional[float] = None
    body_type: Optional[str] = None
    fuel_type: Optional[str] = None
    transmission: Optional[str] = None
    has_accidents: Optional[bool] = None
    frame_damaged: Optional[bool] = None
    full: Optional[str] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    id: Optional[int] = None

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"

    # ---- Day 5+ prompt-building (mirrors course's Item.make_prompts) ----
    def make_prompts(self, tokenizer, max_tokens, do_round=True):
        tokens = tokenizer.encode(self.summary, add_special_tokens=False)
        if len(tokens) > max_tokens:
            summary = tokenizer.decode(tokens[:max_tokens]).rstrip()
        else:
            summary = self.summary
        self.prompt = f"{QUESTION}\n\n{summary}\n\n{PREFIX}"
        self.completion = f"{round(self.price)}.00" if do_round else str(self.price)

    def to_datapoint(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion}

    @staticmethod
    def push_to_hub(dataset_name: str, train: list["Car"], val: list["Car"], test: list["Car"]):
        DatasetDict({
            "train": Dataset.from_list([c.model_dump() for c in train]),
            "validation": Dataset.from_list([c.model_dump() for c in val]),
            "test": Dataset.from_list([c.model_dump() for c in test]),
        }).push_to_hub(dataset_name)

    @staticmethod
    def push_prompts_to_hub(dataset_name: str, train: list["Car"], val: list["Car"], test: list["Car"]):
        DatasetDict({
            "train": Dataset.from_list([c.to_datapoint() for c in train]),
            "val": Dataset.from_list([c.to_datapoint() for c in val]),
            "test": Dataset.from_list([c.to_datapoint() for c in test]),
        }).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )