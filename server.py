from flask import Flask, request, jsonify
import os
import torch
import numpy as np
import cv2
from pyngrok import ngrok  # ngrok

app = Flask(__name__)

YOLOV5_DIR = 'C:/Users/mavuo/PycharmProjects/esp32camchinh/yolov5'
YOLO_WEIGHTS = 'C:/Users/mavuo/PycharmProjects/esp32camchinh/yolov5/yolov5s.pt'

# Biến global để lưu mô hình
model = None

def load_model():
    """Tải mô hình YOLOv5 một lần duy nhất"""
    global model
    model = torch.hub.load(
        YOLOV5_DIR, 'custom',
        path=YOLO_WEIGHTS,
        source='local',
        force_reload=False
    )
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.conf = 0.25

def found_person(img_bytes):
    """Phát hiện người trong ảnh sử dụng mô hình đã tải sẵn"""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model(img)
    predictions = results.xyxy[0].cpu().numpy()
    persons = [pred for pred in predictions if int(pred[5]) == 0]
    return len(persons) > 0

@app.route('/gi_cung_Toan', methods=['POST'])
def detect():
    if 'id' not in request.form or 'image' not in request.files:
        return jsonify({'error': 'Missing id or image'}), 400

    esp32_id = request.form['id']
    img_bytes = request.files['image'].read()
    person = 1 if found_person(img_bytes) else 0

    return jsonify({
        'id': esp32_id,
        'person': person,
        'message': 'có người trong lớp học' if person else 'không có người trong lớp học'
    })

if __name__ == '__main__':
    load_model()
    print("Model loaded successfully!")

    # Sử dụng miền cố định với tham số domain
    public_url = ngrok.connect(5000, domain="terribly-stable-starling.ngrok-free.app")
    print(f" * Ngrok tunnel: {public_url}")

    app.run(host='0.0.0.0', port=5000, threaded=True)
