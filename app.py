from flask import Flask, render_template, request
import cv2
import dlib
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Khởi tạo mô hình nhận diện khuôn mặt
detector = dlib.get_frontal_face_detector()

# Load mô hình nhận diện độ tuổi và giới tính
age_gender_model = load_model("model_age.hdf5")

# Route trang chủ
@app.route('/')
def index():
    return render_template('index_with_age_gender.html')

# Xử lý yêu cầu upload ảnh và nhận diện khuôn mặt, độ tuổi, giới tính
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index_with_age_gender.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index_with_age_gender.html', error='No selected file')

    if file:
        # Đọc ảnh và nhận diện khuôn mặt
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        faces = detector(image)

        # Vẽ đường biên xung quanh khuôn mặt và thêm thông tin độ tuổi, giới tính
        for face in faces:
            cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

            # Chuẩn bị ảnh để đưa vào mô hình
            face_roi = cv2.resize(image[face.top():face.bottom(), face.left():face.right()], (64, 64))
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)  # Chuyển đổi sang định dạng màu RGB
            face_roi = cv2.resize(face_roi, (50, 50))  # Thay đổi kích thước ảnh
            face_roi = np.expand_dims(face_roi, axis=0)

            # Đưa ảnh vào mô hình
            predictions = age_gender_model.predict(face_roi)
            age_probabilities = predictions[0]
            age = int(np.argmax(age_probabilities))  # Lấy giá trị độ tuổi có xác suất cao nhất
            # gender = "Male" if predictions[1].any() < 0.5 else "Female"
            gender_prob = predictions[0]  # Lấy giá trị giới tính
            gender = "Male" if gender_prob < 0.5 else "Female"

            # Hiển thị thông tin độ tuổi và giới tính trên ảnh
            cv2.putText(image, f'Age: {age}', (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, f'Gender: {gender}', (face.left(), face.top() + face.height() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Lưu ảnh đã nhận diện
        cv2.imwrite('static/result.jpg', image)

        return render_template('index_with_age_gender.html', result='static/result.jpg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)






# # Import các thư viện cần thiết
# from flask import Flask, render_template, request
# import cv2
# import dlib
# import numpy as np

# app = Flask(__name__)

# # Khởi tạo mô hình nhận diện khuôn mặt
# detector = dlib.get_frontal_face_detector()

# # Route trang chủ
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Xử lý yêu cầu upload ảnh và nhận diện khuôn mặt
# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return render_template('index.html', error='No file part')

#     file = request.files['file']

#     if file.filename == '':
#         return render_template('index.html', error='No selected file')

#     if file:
#         # Đọc ảnh và nhận diện khuôn mặt
#         image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
#         faces = detector(image)

#         # Vẽ đường biên xung quanh khuôn mặt
#         for face in faces:
#             cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

#         # Lưu ảnh đã nhận diện
#         cv2.imwrite('static/result.jpg', image)

#         return render_template('index.html', result='static/result.jpg')

# if __name__ == '__main__':
#     # Chỉnh cổng mặc định thành 80 để có thể truy cập từ xa
#     app.run(host='0.0.0.0', port=8080, debug=True)



# # # Import các thư viện cần thiết
# # from flask import Flask, render_template, request
# # import cv2
# # import dlib
# # import numpy as np

# # app = Flask(__name__)

# # # Khởi tạo mô hình nhận diện khuôn mặt
# # detector = dlib.get_frontal_face_detector()

# # # Route trang chủ
# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # # Xử lý yêu cầu upload ảnh và nhận diện khuôn mặt
# # @app.route('/upload', methods=['POST'])
# # def upload():
# #     if 'file' not in request.files:
# #         return render_template('index.html', error='No file part')

# #     file = request.files['file']

# #     if file.filename == '':
# #         return render_template('index.html', error='No selected file')

# #     if file:
# #         # Đọc ảnh và nhận diện khuôn mặt
# #         image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
# #         faces = detector(image)

# #         # Vẽ đường biên xung quanh khuôn mặt
# #         for face in faces:
# #             cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

# #         # Lưu ảnh đã nhận diện
# #         cv2.imwrite('static/result.jpg', image)

# #         return render_template('index.html', result='static/result.jpg')

# # if __name__ == '__main__':
# #     app.run(debug=True)
