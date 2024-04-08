from flask import Flask, jsonify, request, render_template
from utils import *

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
	return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
	if request.method == 'POST':
		file = request.files['file']
		image_byte = file.read()
		prediction = predict_class(image_byte)
		return jsonify({'prediction': prediction}) 
		
	else:
		return "Please upload an image"


if __name__ == '__main__':
	app.run(debug=True)