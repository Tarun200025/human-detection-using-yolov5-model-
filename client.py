import requests

# Replace with your Flask server URL and port
url = "http://127.0.0.1:9000/detect"

# JSON data must include the RTSP URL of the camera
data = {"rtsp_url": "rtsp://admin:Admin@123@192.168.10.101:554/Streaming/Channels/201"}

try:
    response = requests.post(url, json=data)
    print("Status code:", response.status_code)
    print("Response text:", response.text)  # raw response

    # Only parse JSON if response has content
    if response.text:
        try:
            print("JSON:", response.json())
        except Exception as e:
            print("JSON decode error:", e)

except requests.exceptions.RequestException as e:
    print("Request failed:", e)


