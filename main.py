from flask import Flask, request, Response
from segmentation import get_yolov5, get_image_from_bytes
from flask_cors import CORS
import io
from PIL import Image
import json

model = get_yolov5()

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def hello():
    return {'msg': 'hello'}

@app.route('/notify/v1/health', methods=['GET'])
def get_health():
    return {'msg': 'OK'}

@app.route("/object-to-json", methods=['POST'])
def detect_object_return_json_result():
    file = request.files['file'].read()
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    return {"result": detect_res}

@app.route("/object-to-img", methods=['POST'])
def detect_object_return_base64_img():
    file = request.files['file'].read()
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()  # updates results.imgs with boxes and labels
    for img in results.imgs:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
    return Response(response=bytes_io.getvalue(), mimetype="image/jpeg")

if __name__ == '__main__':
    app.run(debug=True)
