import cv2
import numpy as np

def get_traffic_light_color(roi):
    # Convert the region of interest (ROI) to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define color ranges for red, yellow, and green in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])

    # Create masks for each color
    mask_red = cv2.inRange(hsv_roi, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)

    # Count the number of non-zero pixels in each mask
    count_red = np.count_nonzero(mask_red)
    count_yellow = np.count_nonzero(mask_yellow)
    count_green = np.count_nonzero(mask_green)

    # Determine the color based on the maximum count
    if count_red > count_yellow and count_red > count_green:
        return "Red"
    elif count_yellow > count_red and count_yellow > count_green:
        return "Yellow"
    else:
        return "Green"

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

# Open a video capture
cap = cv2.VideoCapture(0)  # 0 for default camera, you can also provide the video file path

while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width, _ = frame.shape

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Post-process the output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 9:  # 9 corresponds to the class "traffic light" in the COCO dataset
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Extract the region of interest (ROI) around the traffic light
                roi = frame[y:y+h, x:x+w]

                # Get the color of the traffic light
                color = get_traffic_light_color(roi)

                # Draw bounding box and display the color
                color_text = f"Color: {color}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, color_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Traffic Light Detection", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

