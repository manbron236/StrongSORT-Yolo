import cv2
import torch
from strong_sort import StrongSORT
from yolov5.models.common import DetectMultiBackend
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import sys
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import warnings
from pathlib import Path
import os

warnings.filterwarnings('ignore')

# 현재 파일 위치를 기준으로 yolov5와 strong_sort 폴더를 경로에 추가
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 프로젝트 루트 디렉토리
YOLOV5_PATH = ROOT / 'yolov5'
STRONGSORT_PATH = ROOT / 'strong_sort'
UTILS_PATH = YOLOV5_PATH / 'utils'

# 경로 설정
sys.path.extend([str(ROOT), str(YOLOV5_PATH), str(STRONGSORT_PATH), str(UTILS_PATH)])

class DetectionSystem:
    def __init__(self, model_path, device='cuda'):
        # YOLO 모델 초기화
        self.model = DetectMultiBackend(weights=model_path, device=device)
        self.model.warmup()
        self.sort_tracker = StrongSORT(
            model_weights='weights/strong_sort_weights.pt',
            device=device
        )

        # 카운팅 박스와 라인 설정 (임의로 설정된 좌표)
        self.counting_box = (100, 100, 500, 400)  # (x1, y1, x2, y2) 형식의 박스 좌표
        self.entry_line_y = 80   # 상단에 entry 라인 설정
        self.exit_line_y = 420   # 하단에 exit 라인 설정

    def detect_and_track(self, frame):
        # YOLO 모델을 사용하여 객체 탐지 수행
        results = self.model(frame, augment=False, visualize=False)
        detections = results.xyxy[0].cpu().numpy()

        counts = {'entry': 0, 'stay': 0, 'exit': 0}
        tracked_objects = self.sort_tracker.update(
            bbox_xywh=[(det[0], det[1], det[2], det[3]) for det in detections],
            confidences=[det[4] for det in detections],
            classes=[int(det[5]) for det in detections],
            ori_img=frame
        )

        for *xyxy, track_id, class_id, conf in tracked_objects:
            if class_id == 0:  # 사람이 탐지된 경우 ('0' 클래스가 '사람'으로 설정)
                self.apply_mosaic(frame, xyxy)

                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if self.counting_box[0] < cx < self.counting_box[2] and self.counting_box[1] < cy < self.counting_box[3]:
                    counts['stay'] += 1

                if cy < self.entry_line_y:
                    counts['entry'] += 1
                elif cy > self.exit_line_y:
                    counts['exit'] += 1

        return frame, counts

    @staticmethod
    def apply_mosaic(frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        face_region = frame[y1:y2, x1:x2]
        face_region = cv2.resize(face_region, (10, 10), interpolation=cv2.INTER_LINEAR)
        frame[y1:y2, x1:x2] = cv2.resize(face_region, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

class MainWindow(QMainWindow):
    def __init__(self, detection_system):
        super().__init__()
        self.setWindowTitle("Detection and Counting System")

        self.video_label = QLabel(self)
        self.count_label = QLabel("Entry: 0 | Stay: 0 | Exit: 0", self)
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.count_label)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.detection_system = detection_system
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.entry_count = 0
        self.stay_count = 0
        self.exit_count = 0

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame, counts = self.detection_system.detect_and_track(frame)
            self.entry_count += counts['entry']
            self.stay_count += counts['stay']
            self.exit_count += counts['exit']

            self.count_label.setText(f"Entry: {self.entry_count} | Stay: {self.stay_count} | Exit: {self.exit_count}")

            qt_image = self.convert_cv_to_qt(frame)
            self.video_label.setPixmap(qt_image)

    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        event.accept()

    @staticmethod
    def convert_cv_to_qt(cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    detection_system = DetectionSystem(model_path="weights/crowdhuman_yolov5m.pt")
    main_window = MainWindow(detection_system)
    main_window.show()
    sys.exit(app.exec_())
