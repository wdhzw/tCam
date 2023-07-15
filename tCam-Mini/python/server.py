from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
from matplotlib import cm
import base64
import numpy as np
import cv2
import logging
import datetime

app = Flask(__name__)
CORS(app)  # enable CORS

# To store data between requests
current_image_data = None
current_result = None
current_confidence = None
# counter to keep track of requests
# counter = 0
# # Setting up the logging configuration
# logging.basicConfig(filename='server_log.log', level=logging.INFO, 
#                     format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

@app.route('/')
def home():
    return render_template('stream.html')

@app.route('/postdata', methods=['POST'])
def postdata():
    global current_image_data, current_result, current_confidence, counter
    data = request.get_json()

    if 'result' in data and 'confidence' in data and 'image_data' in data:
        # ESP32 is sending data, store it
        current_result = data.get('result')
        current_confidence = data.get('confidence')
        current_image_data = data.get('image_data')

        # Get the current time
        current_time = datetime.datetime.now()

        # Write the time, image data, and result to a file
        with open('recorded_data.txt', 'a') as f:
            f.write(f'Time: {current_time}, Result: {current_result}, Confidence: {current_confidence}\n')

        # decode base64 string to raw image data and reshape it to its original shape
        imgdata = base64.b64decode(current_image_data)
        image_np = np.frombuffer(imgdata, dtype=np.uint16).reshape((120, 160))

        # normalize to 8-bit range
        image_np = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # apply colormap (optional)
        image_np = cv2.applyColorMap(image_np, cv2.COLORMAP_JET)

        # convert the image to a format that can be sent as JSON
        _, img_encoded = cv2.imencode('.png', image_np)
        current_image_data = base64.b64encode(img_encoded.tostring()).decode('utf-8')

        """
        # Increment counter and log every nth request
        counter += 1
        if counter % 1 == 0:  # replace 'n' with desired interval
            # Logging the result and confidence
            logging.info(f'Result: {current_result}, Confidence: {current_confidence}')
        """
    response_data = {
        "status": "ok",
        "result": current_result,
        "confidence": current_confidence,
        "image_data": current_image_data
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='192.168.1.111', port=5000)
