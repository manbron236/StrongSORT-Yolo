import os
import sys
import numpy as np
import cv2
import torch
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

# YOLOv5와 StrongSORT 경로 추가
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # StrongSORT-YOLO root directory

if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort to PATH

# Now you can import YOLOv5 and StrongSORT modules
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_file
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from strong_sort.strong_sort import StrongSORT
from strong_sort.utils.parser import get_config

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Person Detection with Mosaic Masking")
        self.setGeometry(100, 100, 800, 600)

        # Initialize PyQt layout
        self.label = QLabel(self)
        self.label.resize(640, 480)
        
        # Stream control buttons
        self.start_button = QPushButton("Start Stream", self)
        self.start_button.clicked.connect(self.start_stream)
        
        self.stop_button = QPushButton("Stop Stream", self)
        self.stop_button.clicked.connect(self.stop_stream)
        self.stop_button.setEnabled(False)
        
        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        
        # Main widget setup
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Streaming setup
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None  # Video capture object

        # Model and tracking configuration
        self.yolo_weights = "weights/crowdhuman_yolov5m.pt"
        self.strong_sort_weights = "weights/osnet_x0_25_msmt17.pt"
        self.config_strongsort = "strong_sort/configs/strong_sort.yaml"
        self.device = select_device('')

    def start_stream(self):
        self.cap = cv2.VideoCapture(0)  # Use webcam (source=0)
        self.model = DetectMultiBackend(self.yolo_weights, device=self.device)
        stride, self.names, _ = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size((640, 640), s=stride)
        
        # StrongSORT setup
        cfg = get_config()
        cfg.merge_from_file(self.config_strongsort)
        self.strong_sort = StrongSORT(self.strong_sort_weights, self.device, max_dist=cfg.STRONGSORT.MAX_DIST)
        
        # Start timer and buttons
        self.timer.start(30)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_stream(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.label.clear()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        # Image processing and model inference
        im = torch.from_numpy(frame).to(self.device)
        im = im.float() / 255.0
        im = im.permute(2, 0, 1).unsqueeze(0)
        
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45, classes=[0], max_det=1000)
        
        for det in pred:
            annotator = Annotator(frame, line_width=2, pil=False)
            if det is not None and len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], frame.shape).round()
                xywhs = det[:, 0:4]
                confs = det[:, 4]
                clss = det[:, 5]

                # Pass detections to StrongSORT
                outputs = self.strong_sort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
                
                if len(outputs) > 0:
                    for output in outputs:
                        *xyxy, track_id, cls = output[:6]
                        x1, y1, x2, y2 = map(int, xyxy)
                        label = f'{track_id} {self.names[int(cls)]}'
                        
                        # Apply face mosaic
                        frame = self.face_mosaic(frame, x1, y1, x2 - x1, y2 - y1)
                        
                        # Annotate bounding box with label
                        annotator.box_label(xyxy, label, color=colors(cls, True))
            else:
                self.strong_sort.increment_ages()

        frame = annotator.result()
        
        # Convert image to QImage and display
        qimg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_BGR888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def face_mosaic(self, image, x, y, w, h, ratio=0.1):
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
        face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = face
        return image

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
