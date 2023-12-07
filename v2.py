import cv2
import numpy as np

# Load pre-trained YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

# Thêm các loại phương tiện mới vào danh sách classes
additional_classes = ["bicycle", "car", "motorbike", "bus", "truck"]
classes.extend(additional_classes)

# Thay đổi cách truy cập layer_names
layer_names = net.getUnconnectedOutLayersNames()

# Function to detect traffic congestion
def detect_traffic_congestion(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to read image from {image_path}")
        return

    height, width, channels = img.shape

    # Detecting objects in the image
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    # Sử dụng layer_names trực tiếp
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Processing the detected objects
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.1 and classes[class_id] in additional_classes:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    # Count the number of vehicles
    vehicle_count = len(indexes)

    # Determine traffic congestion based on the number of vehicles
    if vehicle_count < 10:
        congestion_status = "Traffic is flowing smoothly."
    elif vehicle_count >= 10 and vehicle_count < 20:
        congestion_status = "Traffic is moderate."
    else:
        congestion_status = "Traffic is congested."

    # Drawing bounding boxes around detected vehicles
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the result
    cv2.imshow("Traffic Congestion Detection", img)
    print(congestion_status)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
# detect_traffic_congestion("traffic_image.jpg")

# Example usage
detect_traffic_congestion("test_img/2.png")
