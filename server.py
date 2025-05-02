from flask import Flask, request, jsonify
import os
import torch
import numpy as np
import cv2

app = Flask(__name__)

YOLOV5_DIR = 'yolov5'
YOLO_WEIGHTS = 'yolov5s.pt'

# Biến global để lưu mô hình
model = None


def load_model():
    """Tải mô hình YOLOv5 một lần duy nhất"""
    global model
    # Sử dụng torch.hub để tải mô hình từ YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=os.path.join(YOLOV5_DIR, YOLO_WEIGHTS),
                           force_reload=False)
    # Chỉ định thiết bị GPU nếu có
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    # Đặt độ tin cậy tối thiểu
    model.conf = 0.25


def found_person(img_bytes):
    """Phát hiện người trong ảnh sử dụng mô hình đã tải sẵn"""
    # Đọc ảnh từ bytes
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Thực hiện dự đoán trực tiếp mà không cần lưu file
    results = model(img)

    # Kiểm tra xem có class 0 (người) trong kết quả không
    predictions = results.xyxy[0].cpu().numpy()  # Lấy kết quả dạng tensor

    # Tìm các đối tượng là người (class 0)
    persons = [pred for pred in predictions if int(pred[5]) == 0]

    return len(persons) > 0


@app.route('/detect', methods=['POST'])
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
    # Tải mô hình khi khởi động server
    load_model()
    print("Model loaded successfully!")
    # Khởi động server
    app.run(host='0.0.0.0', port=5000, threaded=True)
