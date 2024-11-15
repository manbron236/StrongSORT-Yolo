import threading
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from datetime import datetime
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize the main application window
root = tk.Tk()
root.title("YOLO Zone Management with Counting")

# Create a frame for the video display
video_frame = tk.Frame(root, width=600, height=400)
video_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Create a canvas for the video feed
video_label = tk.Canvas(video_frame, width=600, height=400, bg="black")
video_label.pack()

# Create a frame for controls
control_frame = tk.Frame(root)
control_frame.pack(side=tk.RIGHT, padx=10, pady=30, fill=tk.Y)

# Create a table for counting results
table_frame = tk.Frame(root)
table_frame.pack(side=tk.BOTTOM, padx=10, pady=10, fill=tk.X)

columns = ("date", "time", "entry", "stay", "exit", "total")
table = ttk.Treeview(table_frame, columns=columns, show="headings", height=5)
for col in columns:
    table.heading(col, text=col.capitalize())
table.pack(fill=tk.X)

# Load YOLO model and DeepSort tracker
# Load YOLO model and set device to GPU
model = YOLO('yolov8n.pt')  # YOLO 모델 로드
model.to("cuda")  # GPU로 설정

# DeepSort 초기화 시 GPU 설정
tracker = DeepSort(max_age=30, n_init=3)

# Shared global variables and lock
frame = None
entry_count, stay_count, exit_count = 0, 0, 0
lock = threading.Lock()

# Drawing data structures for lines and boxes
lines = {"entry": [], "exit": []}  # Entry and exit lines
boxes = []  # Counting boxes
current_object = None

def camera_thread():
    """Capture frames from the RTSP stream or webcam."""
    global frame
    rtsp_url = "rtsp://admin:Bora7178@dev99ok.iptime.org:1038/profile2/media.smp"
    
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)  # RTSP 스트림에 맞게 설정
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Failed to retrieve frame. Retrying...")
            cap = cv2.VideoCapture(rtsp_url)
            continue
        with lock:
            frame = cv2.resize(img, (600, 400))
        time.sleep(0.03)

def model_thread():
    """Run YOLO detection and DeepSort tracking."""
    global entry_count, stay_count, exit_count, frame
    while True:
        with lock:
            if frame is None:
                continue
            img = frame.copy()

        results = model(img)
        detections = []
        for result in results:
            for id, box in enumerate(result.boxes.xyxy):
                cls = int(result.boxes.cls[id])
                conf = result.boxes.conf[id]
                if cls == 0:  # 'person' class only
                    x1, y1, x2, y2 = map(int, box)
                    detections.append(([x1, y1, x2, y2], conf))

        tracks = tracker.update_tracks(detections, frame=img)
        current_stay_count = 0
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltwh())
            print(f"Tracking ID {track_id}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            center = get_center((x1, y1, x2, y2))

            # Check entry line crossing
            for line in lines["entry"]:
                if check_line_crossing(center, line):
                    entry_count += 1

            # Check stay in counting box
            for box in boxes:
                if check_in_box(center, box):
                    current_stay_count += 1

            # Check exit line crossing
            for line in lines["exit"]:
                if check_line_crossing(center, line):
                    exit_count += 1

        with lock:
            stay_count = current_stay_count
        time.sleep(0.03)

def ui_thread():
    """Update the UI and display frames with detections."""
    global frame
    while True:
        with lock:
            if frame is None:
                continue
            img = frame.copy()

        # Convert frame to Tkinter format
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        
        video_label.create_image(0, 0, anchor=tk.NW, image=img_tk)
        video_label.imgtk = img_tk  # Keep reference to avoid garbage collection

        redraw_objects()  # Draw lines and boxes on the frame
        root.update_idletasks()
        time.sleep(0.03)


def get_center(box):
    """Calculate the center of a bounding box."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx, cy


def check_line_crossing(center, line):
    """Check if a point crosses a line."""
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    if y1 == y2:  # Horizontal line
        return (center[1] > y1) if line.get("direction", "down") == "down" else (center[1] < y1)
    return False


def check_in_box(center, box):
    """Check if a point is inside a box."""
    x1, y1 = box["x1"], box["y1"]
    x2, y2 = box["x2"], box["y2"]
    return x1 <= center[0] <= x2 and y1 <= center[1] <= y2


def update_table():
    """Update the table every second with the current count data."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    with lock:
        table.insert(
            "", tk.END,
            values=(date_str, time_str, entry_count, stay_count, exit_count, entry_count)
        )
    root.after(1000, update_table)

# 선을 그리는 함수
def start_draw(event):
    global current_object
    current_object = {"start": (event.x, event.y), "end": None}

def draw_object(event):
    global current_object
    if current_object:
        current_object["end"] = (event.x, event.y)
        threading.Thread(target=draw_in_background).start()

def finish_draw(event):
    global current_object
    if current_object and current_object.get("end"):
        x1, y1 = current_object["start"]
        x2, y2 = event.x, event.y
        if draw_mode.get() == "line":
            if line_type.get() == "entry":
                lines["entry"].append({"start": (x1, y1), "end": (x2, y2)})
            elif line_type.get() == "exit":
                lines["exit"].append({"start": (x1, y1), "end": (x2, y2)})
        elif draw_mode.get() == "box":
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    current_object = None

# 별도의 스레드에서 선을 그리는 함수
def draw_in_background():
    root.after(0, redraw_objects)

def redraw_objects():
    """Redraw all lines and boxes on the canvas."""
    video_label.delete("object")  # Clear all existing objects
    for obj in lines["entry"]:
        x1, y1 = obj["start"]
        x2, y2 = obj["end"]
        video_label.create_line(x1, y1, x2, y2, fill="blue", width=2, tags="object")
    for obj in lines["exit"]:
        x1, y1 = obj["start"]
        x2, y2 = obj["end"]
        video_label.create_line(x1, y1, x2, y2, fill="green", width=2, tags="object")
    for obj in boxes:
        x1, y1 = obj["x1"], obj["y1"]
        x2, y2 = obj["x2"], obj["y2"]
        video_label.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="object")

# Initialize Tkinter UI elements for drawing mode and line type
draw_mode = tk.StringVar(value="line")
line_type = tk.StringVar(value="entry")

tk.Label(control_frame, text="Drawing Mode").pack()
tk.Radiobutton(control_frame, text="Line", variable=draw_mode, value="line").pack()
tk.Radiobutton(control_frame, text="Box", variable=draw_mode, value="box").pack()
tk.Label(control_frame, text="Line Type").pack()
tk.Radiobutton(control_frame, text="Entry Line", variable=line_type, value="entry").pack()
tk.Radiobutton(control_frame, text="Exit Line", variable=line_type, value="exit").pack()

video_label.bind("<Button-1>", start_draw)
video_label.bind("<B1-Motion>", draw_object)
video_label.bind("<ButtonRelease-1>", finish_draw)

# Start threads
camera_thread = threading.Thread(target=camera_thread, daemon=True)
model_thread = threading.Thread(target=model_thread, daemon=True)
ui_thread = threading.Thread(target=ui_thread, daemon=True)
camera_thread.start()
model_thread.start()
ui_thread.start()

# Schedule the table update loop
root.after(1000, update_table)
root.mainloop()
