import requests
import numpy as np

URL = "http://127.0.0.1:5000/predict"

# Our BEAT_SIZE = 140
beat_length = 140

# For now, send a random beat (later you can replace with a real one)
dummy_beat = np.random.randn(beat_length).tolist()

payload = {"beat": dummy_beat}

response = requests.post(URL, json=payload)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
