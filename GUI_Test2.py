import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

# Initialize the main application window
root = tk.Tk()
root.title("YOLO Zone Management")

# Create a frame for the video display
video_frame = tk.Frame(root, width=600, height=400)
video_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Create a canvas for the video feed
video_label = tk.Canvas(video_frame, width=600, height=400, bg="black")
video_label.pack()

# Create a frame for controls
control_frame = tk.Frame(root)
control_frame.pack(side=tk.RIGHT, padx=10, pady=30, fill=tk.Y)

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

# Data structures to store lines and boxes
lines = {"entry": [], "exit": []}  # Entry and exit lines
boxes = []  # Counting boxes
current_object = None  # Currently being drawn object


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


def redraw_objects():
    """Redraw all lines and boxes."""
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


def process_video():
    """Process video feed and run YOLO detection."""
    cap = cv2.VideoCapture(0)  # Webcam feed
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for display
        frame = cv2.resize(frame, (600, 400))

        # Get confidence threshold
        conf_threshold = confidence_slider.get() / 100.0

        # Run YOLO detection
        results = model(frame)

        # Filter and display detection results
        for result in results:
            for id, box in enumerate(result.boxes.xyxy):
                cls = int(result.boxes.cls[id])
                conf = result.boxes.conf[id]
                if cls == 0 and conf >= conf_threshold:  # 'person' class only
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame to Tkinter format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.create_image(0, 0, anchor=tk.NW, image=imgtk)
        video_label.imgtk = imgtk

        redraw_objects()  # Redraw all objects on top of the video frame
        root.update()

# Bind mouse events to the video label
video_label.bind("<Button-1>", start_draw)  # Left mouse click to start drawing
video_label.bind("<B1-Motion>", draw_object)  # Drag to draw
video_label.bind("<ButtonRelease-1>", finish_draw)  # Release mouse to finish drawing

# Start the video processing loop
root.after(0, process_video)

# Run the GUI loop
root.mainloop()
