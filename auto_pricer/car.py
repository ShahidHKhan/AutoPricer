from pydantic import BaseModel
from typing import Optional


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
    id: Optional[int] = None

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"

    @classmethod
    def from_hub(cls, repo_id: str) -> tuple[list["Car"], list["Car"], list["Car"]]:
        from datasets import load_dataset
        dataset = load_dataset(repo_id)

        def to_cars(split):
            return [cls(**{k: (None if v != v else v) for k, v in row.items()}) for row in split]

        return to_cars(dataset["train"]), to_cars(dataset["validation"]), to_cars(dataset["test"])