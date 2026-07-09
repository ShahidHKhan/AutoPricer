import modal
from modal import Image

app = modal.App("hello")
image = Image.debian_slim().pip_install("requests")

@app.function(image=image)
def hello() -> str:
    import requests
    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    city, region, country = data["city"], data["region"], data["country"]
    return f"Hello from {city}, {region}, {country}!!"