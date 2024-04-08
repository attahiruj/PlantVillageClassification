import requests

# Read the image file
url = 'http://127.0.0.1:5000/predict'
img_path = 'data/plantvillage/Tomato___Septoria_leaf_spot/01f54ad9-9c03-4ffd-86f4-829fc2939120___Matt.S_CG 0702.JPG'

resp = requests.post(
                        "http://localhost:5000/predict",
                        files={"file": open(img_path, 'rb')}
                    )

print(resp.text)