from flask import Flask, render_template, request
import cv2
import dlib
import numpy as np
from flask_bootstrap import Bootstrap

# khởi tạo nền web trên flask với nội dung __name__
app = Flask(__name__)
# xuất hiện phương thức bootstrap
Bootstrap(app)

# Khởi tạo mô hình nhận diện khuôn mặt
detector = dlib.get_frontal_face_detector()

# Route trang chủ
@app.route('/')
def index():
    return render_template('index_bootstrap.html')

# Xử lý yêu cầu upload ảnh và nhận diện khuôn mặt
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index_bootstrap.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index_bootstrap.html', error='No selected file')

    if file:
        # Đọc ảnh và nhận diện khuôn mặt
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        faces = detector(image)

        # Vẽ đường biên xung quanh khuôn mặt
        for face in faces:
            cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

        # Lưu ảnh đã nhận diện
        cv2.imwrite('static/result.jpg', image)

        return render_template('index_bootstrap.html', result='static/result.jpg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
