import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

# Initialize the main application window
root = tk.Tk()
root.title("YOLO Zone Management")

# Create a frame for the video display
video_frame = tk.Frame(root, width=600, height=400)
video_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Create a label for the video feed
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
        video_label.delete("current_draw")  # Remove previous temporary drawing
        if current_object.get("end"):
            x1, y1 = current_object["start"]
            x2, y2 = current_object["end"]
            if draw_mode.get() == "line":  # Draw a line
                video_label.create_line(x1, y1, x2, y2, fill="blue", width=2, tags="current_draw")
            elif draw_mode.get() == "box":  # Draw a box
                video_label.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="current_draw")


def finish_draw(event):
    """Finish drawing and save the object."""
    global current_object
    if current_object and current_object.get("end"):
        x1, y1 = current_object["start"]
        x2, y2 = current_object["end"]
        if draw_mode.get() == "line":  # Save line
            if line_type.get() == "entry":
                lines["entry"].append({"start": (x1, y1), "end": (x2, y2)})
            elif line_type.get() == "exit":
                lines["exit"].append({"start": (x1, y1), "end": (x2, y2)})
        elif draw_mode.get() == "box":  # Save box
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    current_object = None
    video_label.delete("current_draw")  # Clear temporary drawing
    redraw_objects()  # Redraw all objects


def delete_object(event):
    """Delete an object at the clicked location."""
    x, y = event.x, event.y
    for obj in lines["entry"]:
        if is_point_near_line(x, y, obj["start"], obj["end"]):
            lines["entry"].remove(obj)
            redraw_objects()
            return
    for obj in lines["exit"]:
        if is_point_near_line(x, y, obj["start"], obj["end"]):
            lines["exit"].remove(obj)
            redraw_objects()
            return
    for obj in boxes:
        if is_point_in_box(x, y, obj):
            boxes.remove(obj)
            redraw_objects()
            return


def is_point_near_line(px, py, start, end):
    """Check if a point is near a line."""
    x1, y1 = start
    x2, y2 = end
    distance = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    return distance < 5


def is_point_in_box(px, py, box):
    """Check if a point is inside a box."""
    x1, y1 = box["x1"], box["y1"]
    x2, y2 = box["x2"], box["y2"]
    return x1 <= px <= x2 and y1 <= py <= y2


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


# Bind mouse events to the video label
video_label.bind("<Button-1>", start_draw)  # Left mouse click to start drawing
video_label.bind("<B1-Motion>", draw_object)  # Drag to draw
video_label.bind("<ButtonRelease-1>", finish_draw)  # Release mouse to finish drawing
video_label.bind("<Button-3>", delete_object)  # Right mouse click to delete

# Run the GUI loop
root.mainloop()
