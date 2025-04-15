from flask import Flask, render_template, Response, redirect, url_for
import cv2
import numpy as np

app = Flask(__name__)

# YOLO Configuration
MODEL_CONFIG = "yolov4.cfg"  # YOLOv4 configuration file
MODEL_WEIGHTS = "yolov4.weights"  # YOLOv4 weights file
COCO_NAMES = "coco.names"  # Class labels

# Load class names
with open(COCO_NAMES, "r") as f:
    CLASS_NAMES = f.read().strip().split("\n")

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(MODEL_CONFIG, MODEL_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Global variable for video capture
cap = None

def detect_objects(frame):
    """Detect objects in a video frame."""
    # Resize the frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    output_layers = [layer_names[int(i) - 1] for i in unconnected_out_layers.flatten()]

    detections = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    confidence_threshold = 0.6  # Adjust confidence threshold for better accuracy
    nms_threshold = 0.4  # Adjust NMS threshold for better overlap handling

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:  # Filter by confidence
                box = detection[0:4] * np.array([w, h, w, h])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{CLASS_NAMES[class_ids[i]]}: {confidences[i]:.2f}"

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/start')
def start_feed():
    """Start the video feed."""
    global cap
    cap = cv2.VideoCapture(0)
    return redirect(url_for('video_feed'))


@app.route('/stop')
def stop_feed():
    """Stop the video feed."""
    global cap
    if cap:
        cap.release()
        cap = None
    return redirect(url_for('index'))


@app.route('/video_feed')
def video_feed():
    """Stream the video feed."""
    def generate():
        global cap
        while cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame for object detection
            frame = detect_objects(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)