import requests
import json
import time

ENDPOINT = "https://shahidhkhan--auto-pricer-service-pricingendpoint-web.modal.run/estimate"

text = (
    "2011 BMW 328i xdrive - Cars and Trucks - Florida, New York. $5,000. "
    "Condition Used - Good. 153xxx miles. Car needs nothing. Exterior is 8/10 "
    "Interior is 9/10. All wheel drive. Zero warning lights on the dash, all "
    "services up to date."
)

start = time.time()
response = requests.post(ENDPOINT, json={"text": text}, stream=True)

for line in response.iter_lines():
    if line:
        elapsed = time.time() - start
        event = json.loads(line)
        print(f"[{elapsed:6.1f}s] {event.get('stage')}")