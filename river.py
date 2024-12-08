import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO

class ObjectDetectionApp(QMainWindow):
    def _init_(self):
        super()._init_()
        self.setWindowTitle("Object Detection UI")
        self.setGeometry(100, 100, 800, 600)
        
        # Main widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Layout
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        # Buttons
        self.video_button = QPushButton("Upload and Detect in Recorded Video")
        self.video_button.clicked.connect(self.open_video_file)
        self.layout.addWidget(self.video_button)
        
        # Display area
        self.video_label = QLabel("Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)
        
        # Variables
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.model = YOLO('/content/Plastic-In-River-Detection/runs/detect/train/weights/best.pt')  # Load the YOLO model
        self.model.to('cuda')  # Ensure GPU usage

    def open_video_file(self):
        self.stop_video()
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.webm)", options=options)
        if file_path:
            self.capture = cv2.VideoCapture(file_path)
            self.timer.start(30)

    def stop_video(self):
        if self.capture:
            self.capture.release()
        self.timer.stop()
        self.video_label.clear()

    def update_frame(self):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                frame = self.object_detection(frame)  # Apply YOLO detection
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = frame.shape
                qimage = QImage(frame.data, width, height, channels * width, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qimage))
            else:
                self.stop_video()

    def object_detection(self, frame):
        # Perform YOLO inference
        results = self.model(frame)  # Run inference on the frame
        annotated_frame = results[0].plot()  # Annotate frame with detection results
        return annotated_frame

    def closeEvent(self, event):
        self.stop_video()
        super().closeEvent(event)

if _name_ == "_main_":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())