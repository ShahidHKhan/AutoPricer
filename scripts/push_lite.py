import os, sys, random
import pandas as pd
import re
from dotenv import load_dotenv
from huggingface_hub import login

sys.path.append(".")
from auto_pricer.car import Car

load_dotenv(".env")
login(token=os.getenv("HF_TOKEN"))

print("Loading 50k rows...")
df = pd.read_csv("data/used_cars_data.csv", nrows=50_000, low_memory=False)

dead_columns = [
    "is_certified", "combine_fuel_economy", "vehicle_damage_category",
    "bed", "cabin", "bed_length", "bed_height",
    "is_oemcpo", "isCab", "main_picture_url",
    "listing_id", "sp_id", "sp_name", "dealer_zip",
    "latitude", "longitude", "vin",
]
df = df.drop(columns=dead_columns)
df = df[df["is_new"] == False]

damage_cols = ["salvage", "theft_title", "frame_damaged", "fleet", "has_accidents"]
df[damage_cols] = df[damage_cols].fillna(False)
df["owner_count"] = df["owner_count"].fillna(df["owner_count"].median())

df["torque"] = df["torque"].str.extract(r"([\d.]+)").astype(float)
df["power"]  = df["power"].str.extract(r"([\d.]+)").astype(float)
numeric_cols = ["horsepower", "torque", "power", "highway_fuel_economy", "city_fuel_economy", "engine_displacement"]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

df = df.dropna(subset=["price", "mileage", "year", "make_name", "model_name"])
df = df[df["price"].between(500, 150_000)]
df = df[df["year"] >= 1990]
df = df[df["mileage"] <= 300_000]

def build_full(row):
    desc = str(row["description"]) if pd.notna(row["description"]) else ""
    desc = re.sub(r'\[!@@.*?@@!\]', '', desc).strip()
    desc = desc[:400]
    return (
        f"{row['year']} {row['make_name']} {row['model_name']} {row['trim_name']}\n"
        f"Body: {row['body_type']} | Mileage: {row['mileage']:,.0f} mi\n"
        f"Horsepower: {row['horsepower']} | Fuel: {row['fuel_type']}\n"
        f"Transmission: {row['transmission_display']}\n"
        f"Accidents: {row['has_accidents']} | Frame damaged: {row['frame_damaged']}\n"
        f"Description: {desc}"
    )

df["full"] = df.apply(build_full, axis=1)

cars = []
for _, row in df.iterrows():
    cars.append(Car(
        title=f"{row['year']} {row['make_name']} {row['model_name']}",
        make=row["make_name"], model=row["model_name"],
        trim=row["trim_name"] if pd.notna(row["trim_name"]) else None,
        year=int(row["year"]), mileage=float(row["mileage"]), price=float(row["price"]),
        horsepower=float(row["horsepower"]) if pd.notna(row["horsepower"]) else None,
        body_type=row["body_type"] if pd.notna(row["body_type"]) else None,
        fuel_type=row["fuel_type"] if pd.notna(row["fuel_type"]) else None,
        transmission=row["transmission_display"] if pd.notna(row["transmission_display"]) else None,
        has_accidents=bool(row["has_accidents"]),
        frame_damaged=bool(row["frame_damaged"]),
        full=row["full"], id=len(cars),
    ))

print(f"Cars built: {len(cars)}")

random.seed(42)
random.shuffle(cars)
n = len(cars)
train = cars[:int(n*0.90)]
val   = cars[int(n*0.90):int(n*0.95)]
test  = cars[int(n*0.95):]

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print("Pushing to ShahidHKhan/used_cars_lite ...")
Car.push_to_hub("ShahidHKhan/used_cars_lite", train, val, test)
print("Done!")