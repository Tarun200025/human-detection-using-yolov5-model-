import sys
import cv2
import time
import os
from ultralytics import YOLO
from flask import Flask, jsonify, request
from threading import Thread

print("Python executable:", sys.executable)
print("Python sys.path:", sys.path)

app = Flask(__name__)

# Load YOLO model (only person class)
model = YOLO("yolov5s.pt")
model.classes = [0]  # Only detect person

# Video codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
os.makedirs("frames", exist_ok=True)

ALERT_SOUND = "alert.wav"

# Store last detection info
last_detection = {"status": "none", "video": None, "frames": 0, "time": None}

CAMERA_FPS = 10  # your camera FPS
DURATION = 10    # seconds to record

def process_rtsp(rtsp_url):
    global last_detection
    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return {"status": "error", "message": "Could not open RTSP stream"}

        # Read first frame to get resolution
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {"status": "error", "message": "No frames from RTSP stream"}

        # Check if person exists in first frame
        results = model(frame, verbose=False)
        detected = any(int(box.cls.cpu().numpy()) == 0 for box in results[0].boxes)
        if not detected:
            cap.release()
            return {"status": "none", "message": "No person detected"}

        print("Person detected! Recording 10s video...")

        timestamp = int(time.time())
        video_name = f"person_{timestamp}.mp4"
        out = cv2.VideoWriter(video_name, fourcc, CAMERA_FPS, (frame.shape[1], frame.shape[0]))

        # Record exact number of frames
        total_frames = CAMERA_FPS * DURATION
        frames_captured = 0
        while frames_captured < total_frames:
            ret, frame = cap.read()
            if not ret:
                continue  # skip if frame not received

            results = model(frame, verbose=False)
            person_boxes = [box for box in results[0].boxes if int(box.cls.cpu().numpy()) == 0]
            results[0].boxes = person_boxes

            annotated_frame = results[0].plot()
            out.write(annotated_frame)

            frames_captured += 1

        # Release video
        out.release()
        cap.release()
        print(f"Video saved: {video_name}")

        # Convert video to frames
        vidcap = cv2.VideoCapture(video_name)
        count = 0
        while True:
            ret, frame = vidcap.read()
            if not ret:
                break
            results = model(frame, verbose=False)
            person_boxes = [box for box in results[0].boxes if int(box.cls.cpu().numpy()) == 0]
            results[0].boxes = person_boxes
            annotated_frame = results[0].plot()
            cv2.imwrite(f"frames/frame_{timestamp}_{count:04d}.jpg", annotated_frame)
            count += 1
        vidcap.release()

        # Play alert
        os.system(f"aplay {ALERT_SOUND}")

        # Update last detection info
        last_detection.update({
            "status": "success",
            "video": video_name,
            "frames": count,
            "time": timestamp
        })

        print(f"Detection complete: {video_name}, {count} frames extracted")
        return {"status": "finished", "video": video_name, "frames": count}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.route("/detect", methods=["POST"])
def detect():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    rtsp_url = data.get("rtsp_url")
    if not rtsp_url:
        return jsonify({"error": "Please provide RTSP URL"}), 400

    thread = Thread(target=process_rtsp, args=(rtsp_url,))
    thread.start()
    return jsonify({"status": "started", "message": "Detection started in background"})


@app.route("/status", methods=["GET"])
def status():
    return jsonify(last_detection)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
