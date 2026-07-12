# auto_pricer/agents/extraction.py

import json
from typing import Optional

from pydantic import BaseModel, Field
from litellm import completion

from auto_pricer.car import Car


class ExtractedListing(BaseModel):
    title: str = Field(description="Full listing title, e.g. '2003 Ford Excursion Limited'")
    make: str
    model: str
    trim: Optional[str] = None
    year: int
    mileage: float = Field(description="Odometer reading in miles")
    price: float = Field(description="Asking price in dollars, numeric only")
    horsepower: Optional[float] = None
    body_type: Optional[str] = None
    fuel_type: Optional[str] = None
    transmission: Optional[str] = None
    has_accidents: Optional[bool] = None
    frame_damaged: Optional[bool] = None
    summary: str = Field(description="1-3 sentence plain description of the car's condition and features")


def extract_car_details(listing_text: str) -> ExtractedListing:
    """Gemini structured extraction: raw page text -> ExtractedListing."""
    prompt = f"""Extract structured car listing data from this text. Return ONLY valid JSON matching this schema, no explanation:

{{
  "title": "...", "make": "...", "model": "...", "trim": "...", "year": ...,
  "mileage": ..., "price": ..., "horsepower": ..., "body_type": "...",
  "fuel_type": "...", "transmission": "...", "has_accidents": ..., "frame_damaged": ...,
  "summary": "..."
}}

For has_accidents and frame_damaged: use null unless the text EXPLICITLY states
an accident or frame damage occurred (e.g. "was in an accident", "frame damage",
"salvage title"). Do NOT infer false from phrases like "clean Carfax", "clean title",
or "no issues" — those are not the same claim as "confirmed no accidents". Only set
these to true or false when the text directly addresses accident/damage history.

Listing text:
{listing_text}"""

    response = completion(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)
    return ExtractedListing(**data)


def to_car(extracted: ExtractedListing) -> Car:
    return Car(
        title=extracted.title,
        make=extracted.make,
        model=extracted.model,
        trim=extracted.trim,
        year=extracted.year,
        mileage=extracted.mileage,
        price=extracted.price,
        horsepower=extracted.horsepower,
        body_type=extracted.body_type,
        fuel_type=extracted.fuel_type,
        transmission=extracted.transmission,
        has_accidents=extracted.has_accidents,
        frame_damaged=extracted.frame_damaged,
        summary=extracted.summary,
    )


def format_summary_for_pricing(extracted: ExtractedListing) -> str:
    """Critical: reformat prose extraction into the training-style template
    before pricing. Skipping this caused a ~3x price error in testing
    ($14,872 vs a $9,122 estimate on the same listing — see 8_scraper.ipynb)."""
    category = extracted.body_type or "Vehicle"
    details_parts = [f"{extracted.mileage:,.0f} miles"]
    if extracted.fuel_type:
        details_parts.append(extracted.fuel_type)
    if extracted.transmission:
        details_parts.append(extracted.transmission)
    details = ", ".join(details_parts)

    return (
        f"Title: {extracted.year} {extracted.make} {extracted.model} {extracted.trim or ''}".strip() + "\n"
        f"Category: {category}\n"
        f"Make: {extracted.make}\n"
        f"Description: {extracted.summary}\n"
        f"Details: {details}."
    )