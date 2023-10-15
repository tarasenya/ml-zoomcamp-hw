import requests

url = "http://localhost:3141/predict"
client = {"job": "unknown", "duration": 270, "poutcome": "failure"}
new_client = {"job": "retired", "duration": 445, "poutcome": "success"}

if __name__ == '__main__':
    print(requests.post(url, json=new_client).json())
