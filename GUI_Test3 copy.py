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
root.geometry("1200x800")  # 기본 윈도우 크기 설정
root.state('zoomed')  # 전체 화면으로 설정

# Layout Frames
video_frame = tk.Frame(root)  # 캠 화면을 위한 프레임
video_frame.pack(fill=tk.BOTH, expand=True)

control_frame = tk.Frame(root, bg="lightgray", height=150)  # 버튼들을 위한 프레임
control_frame.pack(fill=tk.X, padx=10, pady=10)

table_frame = tk.Frame(root, bg="white", height=150)  # 테이블을 위한 프레임
table_frame.pack(fill=tk.X, padx=10, pady=10)

# Create a canvas for the video feed
video_label = tk.Canvas(video_frame)
video_label.pack(fill=tk.BOTH, expand=True)

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

# Event objects for thread synchronization
camera_ready = threading.Event()
model_ready = threading.Event()

# Frame processing interval (only process every Nth frame)
process_frame_interval = 2  # N 프레임 중 1 프레임만 처리
frame_count = 0

# Drawing data structures for lines and boxes
lines = {"entry": [], "exit": []}  # Entry and exit lines
boxes = []  # Counting boxes
current_object = None

# Global variable 추가
processed_frame = None  # 모델에서 처리된 프레임

def camera_thread():
    global frame
    rtsp_url = "rtsp://admin:Bora7178@dev99ok.iptime.org:1038/profile2/media.smp"
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while cap.isOpened():
        cap.grab()  # 가장 최신 프레임 읽기
        ret, img = cap.retrieve()
        if not ret:
            print("RTSP stream lost. Reconnecting...")
            time.sleep(1)
            cap = cv2.VideoCapture(rtsp_url)
            continue

        with lock:
            frame = img.copy()
        camera_ready.set()
        time.sleep(0.01)  # 딜레이 최소화


def model_thread():
    global processed_frame, entry_count, stay_count, exit_count, frame_count
    while True:
        camera_ready.wait()  # Wait for a new frame
        with lock:
            if frame is None:
                continue
            img = frame.copy()

        frame_count += 1
        if frame_count % process_frame_interval != 0:
            with lock:
                processed_frame = img
            model_ready.set()  # Notify ui_thread
            continue

        results = model(img)
        detections = []
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                if int(cls) == 0:  # Detect only 'person'
                    x1, y1, x2, y2 = map(int, box)
                    detections.append(([x1, y1, x2, y2], conf))

        tracks = tracker.update_tracks(detections, frame=img)
        current_stay_count = 0
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltwh())
            center = get_center((x1, y1, x2, y2))

            for line in lines["entry"]:
                if check_line_crossing(center, line):
                    entry_count += 1

            for box in boxes:
                if check_in_box(center, box):
                    current_stay_count += 1

            for line in lines["exit"]:
                if check_line_crossing(center, line):
                    exit_count += 1

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            img = apply_mosaic(img, x1, y1, x2, y2)

        with lock:
            stay_count = current_stay_count
            processed_frame = img
        model_ready.set()  # Notify ui_thread
        camera_ready.clear()

def ui_thread():
    while True:
        model_ready.wait()  # Wait for processed frame
        with lock:
            if processed_frame is None:
                continue
            img = processed_frame.copy()

        label_width = video_label.winfo_width()
        label_height = video_label.winfo_height()
        img_height, img_width = img.shape[:2]
        scale = min(label_width / img_width, label_height / img_height)
        resized_img = cv2.resize(img, (int(img_width * scale), int(img_height * scale)))

        frame_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        video_label.create_image(label_width // 2, label_height // 2, anchor=tk.CENTER, image=img_tk)
        video_label.imgtk = img_tk
        redraw_objects()
        root.update_idletasks()
        model_ready.clear()


def apply_mosaic(image, x1, y1, x2, y2):
    """Apply mosaic to a given region with coordinate validation."""
    h, w, _ = image.shape

    # 좌표 유효성 검사 및 클램핑
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    # 유효한 영역인지 확인
    if x1 >= x2 or y1 >= y2:
        print(f"Invalid region for mosaic: ({x1}, {y1}), ({x2}, {y2})")
        return image

    # 모자이크 적용
    sub_img = image[y1:y2, x1:x2]
    sub_img = cv2.resize(sub_img, (10, 10), interpolation=cv2.INTER_LINEAR)
    sub_img = cv2.resize(sub_img, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = sub_img
    return image

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

def get_valid_canvas_area():
    """캠 화면 내에서만 그리기가 가능하도록 경계를 반환."""
    if processed_frame is None:
        return None

    # 캠 화면의 실제 크기
    frame_height, frame_width = processed_frame.shape[:2]

    # Tkinter에서 표시되는 캔버스 크기
    canvas_width = video_label.winfo_width()
    canvas_height = video_label.winfo_height()

    # 비율 계산
    scale = min(canvas_width / frame_width, canvas_height / frame_height)

    # 캠 화면의 유효 영역 계산
    valid_width = int(frame_width * scale)
    valid_height = int(frame_height * scale)

    # 중앙에 배치된 영역
    x_offset = (canvas_width - valid_width) // 2
    y_offset = (canvas_height - valid_height) // 2

    return x_offset, y_offset, valid_width, valid_height

def is_within_valid_area(x, y):
    """주어진 좌표가 캠 화면의 유효 영역 내에 있는지 확인."""
    valid_area = get_valid_canvas_area()
    if valid_area is None:
        return False

    x_offset, y_offset, valid_width, valid_height = valid_area

    # 좌표가 유효 영역 내에 있는지 확인
    return x_offset <= x <= x_offset + valid_width and y_offset <= y <= y_offset + valid_height


def start_draw(event):
    """그리기를 시작합니다."""
    global current_object

    if not is_within_valid_area(event.x, event.y):
        print("그리기 시작 좌표가 캠 화면 바깥입니다.")
        return  # 유효하지 않은 좌표면 종료

    current_object = {"start": (event.x, event.y), "end": None}

def draw_object(event):
    """그리기 중입니다."""
    global current_object

    if current_object and is_within_valid_area(event.x, event.y):
        current_object["end"] = (event.x, event.y)
        threading.Thread(target=draw_in_background).start()

def finish_draw(event):
    """Finish drawing and save the object."""
    global current_object
    if current_object and current_object.get("end"):
        x1, y1 = current_object["start"]
        x2, y2 = event.x, event.y

        # 캠 화면 유효 영역 가져오기
        valid_area = get_valid_canvas_area()
        if valid_area is None:
            print("캠 화면의 유효 영역을 가져올 수 없습니다.")
            return

        x_offset, y_offset, valid_width, valid_height = valid_area

        # 좌표가 유효 영역 내에 있는지 확인
        if not (x_offset <= x1 <= x_offset + valid_width and
                y_offset <= y1 <= y_offset + valid_height and
                x_offset <= x2 <= x_offset + valid_width and
                y_offset <= y2 <= y_offset + valid_height):
            print("그리기 좌표가 캠 화면 바깥입니다. 그리기를 취소합니다.")
            current_object = None
            return

        if draw_mode.get() == "line":
            # 입장선과 퇴장선은 각각 하나만 생성 가능
            if line_type.get() == "entry":
                if len(lines["entry"]) == 0:  # 입장선이 없다면 추가
                    lines["entry"].append({"start": (x1, y1), "end": (x2, y2)})
                else:
                    print("Entry line already exists. Only one allowed.")
            elif line_type.get() == "exit":
                if len(lines["exit"]) == 0:  # 퇴장선이 없다면 추가
                    lines["exit"].append({"start": (x1, y1), "end": (x2, y2)})
                else:
                    print("Exit line already exists. Only one allowed.")
        elif draw_mode.get() == "box":
            # 카운팅 박스는 하나만 생성 가능
            if len(boxes) == 0:
                boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            else:
                print("Counting box already exists. Only one allowed.")

    current_object = None
    redraw_objects()  # 객체를 다시 그립니다.



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
# 선과 박스 삭제 함수
def delete_entry_line():
    """Delete the entry line."""
    lines["entry"].clear()  # 입장선 리스트 초기화
    redraw_objects()  # 다시 그리기

def delete_exit_line():
    """Delete the exit line."""
    lines["exit"].clear()  # 퇴장선 리스트 초기화
    redraw_objects()  # 다시 그리기

def delete_counting_box():
    """Delete the counting box."""
    boxes.clear()  # 카운팅 박스 리스트 초기화
    redraw_objects()  # 다시 그리기

def delete_all():
    """Delete all lines and boxes."""
    lines["entry"].clear()
    lines["exit"].clear()
    boxes.clear()  # 모든 리스트 초기화
    redraw_objects()  # 다시 그리기

# Initialize Tkinter UI elements for drawing mode and line type
draw_mode = tk.StringVar(value="line")
line_type = tk.StringVar(value="entry")

# Control Frame UI
tk.Label(control_frame, text="Drawing Mode", bg="lightgray", font=("Arial", 12)).grid(row=0, column=0, pady=10, sticky="w")
draw_mode = tk.StringVar(value="line")
tk.Radiobutton(control_frame, text="Line", variable=draw_mode, value="line", bg="lightgray").grid(row=0, column=1, sticky="w")
tk.Radiobutton(control_frame, text="Box", variable=draw_mode, value="box", bg="lightgray").grid(row=0, column=2, sticky="w")

tk.Label(control_frame, text="Line Type", bg="lightgray", font=("Arial", 12)).grid(row=1, column=0, pady=10, sticky="w")
line_type = tk.StringVar(value="entry")
tk.Radiobutton(control_frame, text="Entry Line", variable=line_type, value="entry", bg="lightgray").grid(row=1, column=1, sticky="w")
tk.Radiobutton(control_frame, text="Exit Line", variable=line_type, value="exit", bg="lightgray").grid(row=1, column=2, sticky="w")

tk.Label(control_frame, text="Actions", bg="lightgray", font=("Arial", 12)).grid(row=2, column=0, pady=10, sticky="w")
delete_entry_button = tk.Button(control_frame, text="Delete Entry Line", command=delete_entry_line)
delete_entry_button.grid(row=2, column=1, sticky="w")

delete_exit_button = tk.Button(control_frame, text="Delete Exit Line", command=delete_exit_line)
delete_exit_button.grid(row=2, column=2, sticky="w")

delete_box_button = tk.Button(control_frame, text="Delete Counting Box", command=delete_counting_box)
delete_box_button.grid(row=2, column=3, sticky="w")

delete_all_button = tk.Button(control_frame, text="Delete All", command=delete_all)
delete_all_button.grid(row=2, column=4, sticky="w")

confidence_slider = tk.Scale(control_frame, from_=0, to_=100, orient=tk.HORIZONTAL, length=150, label="Confidence")
confidence_slider.set(50)
confidence_slider.grid(row=3, column=0, columnspan=5, pady=10, sticky="we")

video_label.bind("<Button-1>", start_draw)
video_label.bind("<B1-Motion>", draw_object)
video_label.bind("<ButtonRelease-1>", finish_draw)

# Start threads
threading.Thread(target=camera_thread, daemon=True).start()
threading.Thread(target=model_thread, daemon=True).start()
threading.Thread(target=ui_thread, daemon=True).start()

# Schedule the table update loop
root.after(1000, update_table)
root.mainloop()
