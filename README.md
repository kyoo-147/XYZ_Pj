-------------Vesion 2 Update Face Atthendance Recognition - AI-Lab
----Phát Triển: Bùi Minh Cường - NaVin Tech-----
##########################

Mở đầu:
** Phát hiện và nhận diện cá nhân đơn thể hoặc nhiều khuôn mặt cùng lúc từ camera
  Giao diện đăng kí khuôn mặt với Tkinter
  Khi máy ảnh quá gần với khuôn mặt, out of ROI khỏi vùng camera-cảnh báo Out Of Range sẽ xuất hiện
  Tạo cơ sở dữ liệu từ hình ảnh chụp từ camera
  Nhận diện khuôn mặt-phát hiện và nhận diện với mỗi khung frame hình-đối với single face-sẽ chỉ thực thi nhận dạng một khuôn mặt mới xuất hiện để cải thiện FPS
  Sử dụng OT để đạt chất lượng nhận dạng cải thiện FPS cho mọi frame 
  Hỗ trợ show tên bằng tiếng trung quốc

** Về độ chính xác
  Khi sử dụng ngưỡng khoảng cách "0.6" mô hình dlib có thể đạt độ chính xác lên đến ``99.38%`` trên tiêu chuẩn nhận diện khuôn mặt LFW tiêu chuẩn

** Về thuật toán
  Mô hình CNN dựa trên Residual Neural Network(Mạnh nơ-ron hồi quy)
  Đây là mô hình mạng ResNet với 29 lớp tối ưu
  về cơ bản, đây là một phiên bản của ResNet-34 Network từ bài báo Deep Residual Learning cho Image Recognition bởi nhóm He, Zhang, Ren và Sun với một số lớp đã được loại bỏ và số lượng bộ lọc mỗi khung hình được giảm đi một nửa.

** Tổng quan
   Quá trình triển khai nhận dạng khuôn mặt trong dự án, phát hiện và nhận dạng cho mọi frame
    + Không OT, nhận diện từng frame, nhận dạng
    + Với OT, phát hiện khung hình ban đầu, khung tiếp theo, theo dõi trọng tâm-OT được sử dụng
  Sử đụng OT có tiết kiệm được thời gian tính toán bộ mô tả khuôn mặt để cải thiện FPS

** Source Code
** Install package
 # pip install -r requirements.txt
---------------How to run models-------------------------
** Giao diện đăng kí tên người dùng: open cmd in folder save model: run: python get_faces_from_camera_tkinter.py
or
** Giao diện OpenCV - tương tự như bước trên: run: python get_face_from_camera.py
** Train model, trích xuất tính năng và lưu trữ vào features_all.csv: run: python features_extraction_to_csv.py
** Nhận diện khuôn mặt trên thời gian thực: python face_reco_from_camera.py
** Nhận diện trên thời gian thực với FPS tốt hơn: python face_reco_from_camera_single_face.py
** Nhận diện với OT-FPS tốt nhất: python face_reco_from_camera_ot.py

** Sử dụng bộ dò khuôn mặt phía trước Dlib(HOG)
** Khi chạy file train-từ tệp hình ảnh đã lưu ở bước trước trích xuất dữ liệu và các điểm tọa độ trên khuôn mặt và lưu vào file csv
    + kích thước: ``n*129`` n là số hình ảnh*129 có nghĩa là tên khuôn mặt + 128D là tính ăng của khuôn mặt
** Khi chạy chương trình nhận diện-chương trình sẽ so sánh dữ liệu khuôn mặt đã chụp với dữ liệu khuôn mặt được lưu trữ trước đó để tính khoảng cách Euclide
-- sửa đổi cập nhật nhật kí ``logging.basicConfig(level=logging.DEBUG)`` để in thông tin cho mọi khung hình nếu cần, mặc định sẽ là ``logging.INFO``    

"# v2_X_aTt" 
"# v2_Att_dlib_hog" 

<img src="https://i.imgur.com/wT6B70A.png" alt="Alt text">
<img src="https://i.imgur.com/FLpVLtV.png" alt="Alt text">
