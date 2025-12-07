import requests

url = "https://esd-1.onrender.com/predict"

data = {
    "email": "Congratulations! You won a free prize!"
}

response = requests.post(url, json=data)

print(response.json())
