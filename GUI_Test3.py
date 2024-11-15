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

# Drawing mode and type variables
draw_mode = tk.StringVar(value="line")  # Default mode: line
line_type = tk.StringVar(value="entry")  # Default line type: entry line

# Drawing mode selection
tk.Label(control_frame, text="Drawing Mode").pack()
tk.Radiobutton(control_frame, text="Line", variable=draw_mode, value="line").pack()
tk.Radiobutton(control_frame, text="Box", variable=draw_mode, value="box").pack()

# Line type selection
tk.Label(control_frame, text="Line Type").pack()
tk.Radiobutton(control_frame, text="Entry Line", variable=line_type, value="entry").pack()
tk.Radiobutton(control_frame, text="Exit Line", variable=line_type, value="exit").pack()

# Confidence slider
confidence_slider = tk.Scale(control_frame, from_=0, to_=100, orient=tk.HORIZONTAL, length=200, label="Confidence")
confidence_slider.set(50)  # Default confidence threshold
confidence_slider.pack(pady=10)

# Load YOLO model
model = YOLO('yolov8n.pt')

# Initialize DeepSort
tracker = DeepSort(max_age=30, n_init=3)

# Data structures to store lines and boxes
lines = {"entry": [], "exit": []}  # Entry and exit lines
boxes = []  # Counting boxes
current_object = None  # Currently being drawn object

# Counting variables
entry_count = 0
stay_count = 0
exit_count = 0

def start_draw(event):
    """Start drawing a line or box."""
    global current_object
    current_object = {"start": (event.x, event.y), "end": None}


def draw_object(event):
    """Draw a line or box while dragging."""
    global current_object
    if current_object:
        # Update the current object
        current_object["end"] = (event.x, event.y)
        redraw_objects()  # Redraw all existing objects
        x1, y1 = current_object["start"]
        x2, y2 = event.x, event.y
        if draw_mode.get() == "line":  # Draw a line
            video_label.create_line(x1, y1, x2, y2, fill="blue", width=2, tags="current_draw")
        elif draw_mode.get() == "box":  # Draw a box
            video_label.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="current_draw")

def finish_draw(event):
    """Finish drawing and save the object."""
    global current_object
    if current_object and current_object.get("end"):
        x1, y1 = current_object["start"]
        x2, y2 = event.x, event.y
        if draw_mode.get() == "line":  # Save line
            if line_type.get() == "entry":
                lines["entry"].append({"start": (x1, y1), "end": (x2, y2)})
            elif line_type.get() == "exit":
                lines["exit"].append({"start": (x1, y1), "end": (x2, y2)})
        elif draw_mode.get() == "box":  # Save box
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    current_object = None
    redraw_objects()  # Redraw all objects


# Drawing functions
def redraw_objects():
    """Redraw all lines and boxes."""
    # Clear existing objects before redrawing
    video_label.delete("object")

    # Redraw entry and exit lines
    for obj in lines["entry"]:
        x1, y1 = obj["start"]
        x2, y2 = obj["end"]
        video_label.create_line(x1, y1, x2, y2, fill="blue", width=2, tags="object")
    for obj in lines["exit"]:
        x1, y1 = obj["start"]
        x2, y2 = obj["end"]
        video_label.create_line(x1, y1, x2, y2, fill="green", width=2, tags="object")

    # Redraw counting boxes
    for obj in boxes:
        x1, y1 = obj["x1"], obj["y1"]
        x2, y2 = obj["x2"], obj["y2"]
        video_label.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="object")


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

    # Add the current count data to the table
    table.insert(
        "", tk.END,
        values=(date_str, time_str, entry_count, stay_count, exit_count, entry_count)
    )

    # Schedule the function to run again after 1 second
    root.after(1000, update_table)  # Call update_table every second

def process_video():
    """Process RTSP video feed and run YOLO detection."""
    global entry_count, stay_count, exit_count

    rtsp_url = "rtsp://admin:Bora7178@dev99ok.iptime.org:1038/profile2/media.smp"
    cap = cv2.VideoCapture(rtsp_url)  # RTSP 피드 연결

    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame. Retrying...")
        root.after(1000, process_video)  # Retry after 1 second
        return

    # Resize frame for display
    frame = cv2.resize(frame, (600, 400))

    # YOLO detection 및 DeepSort 추적
    results = model(frame)
    detections = []
    for result in results:
        for id, box in enumerate(result.boxes.xyxy):
            cls = int(result.boxes.cls[id])
            conf = result.boxes.conf[id]
            if cls == 0:  # 'person' class만
                x1, y1, x2, y2 = map(int, box)
                detections.append(([x1, y1, x2, y2], conf))

    # 사람 추적 및 카운트 업데이트
    tracks = tracker.update_tracks(detections, frame=frame)
    current_stay_count = 0  # 현재 체류 인원 초기화
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltwh())
        center = get_center((x1, y1, x2, y2))

        # 입장선 통과 확인
        for line in lines["entry"]:
            if check_line_crossing(center, line):
                entry_count += 1

        # 카운팅박스 안에 있는지 확인
        for box in boxes:
            if check_in_box(center, box):
                current_stay_count += 1

        # 퇴장선 통과 확인
        for line in lines["exit"]:
            if check_line_crossing(center, line):
                exit_count += 1

        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 체류 인원을 업데이트
    global stay_count
    stay_count = current_stay_count

    # Tkinter canvas에 프레임 표시
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.create_image(0, 0, anchor=tk.NW, image=imgtk)
    video_label.imgtk = imgtk

    # Draw objects (lines and boxes) on top of the video frame
    redraw_objects()  # 프레임 위에 객체들을 다시 그려줍니다.

    # Schedule the function to run again after a short delay
    root.after(30, process_video)  # Call process_video every 30 ms

# Drawing functions omitted for brevity (same as before)
# Bind mouse events to the video label
video_label.bind("<Button-1>", start_draw)  # Left mouse click to start drawing
video_label.bind("<B1-Motion>", draw_object)  # Drag to draw
video_label.bind("<ButtonRelease-1>", finish_draw)  # Release mouse to finish drawing

# Start the video processing and table update loops
root.after(0, process_video)
root.after(1000, update_table)

# Run the GUI loop
root.mainloop()
