import requests

# Read the image file
url = 'http://127.0.0.1:5000/predict'
img_path = 'data/test/9.jpeg'

resp = requests.post(
                        "http://localhost:5000/predict",
                        files={"file": open(img_path, 'rb')}
                    )

print(resp.text)