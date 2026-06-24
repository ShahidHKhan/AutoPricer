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