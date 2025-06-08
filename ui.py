import sys
import os
import cv2
import shutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog,
    QLineEdit, QMessageBox, QHBoxLayout, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from yoloFaceCutFace import FaceCropper
import FaceToolApi
from register import FaceRegister
from recognize import FaceRecognizer


class FaceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.cap = None
        self.timer_camera = QTimer()
        self.timer_camera.timeout.connect(self.display_camera_frame)
        self.current_frame = None
        self.recognizer = FaceRecognizer("weights/arcface_model.pth", threshold=0.30)
        self.register_capture_active = False
        self.register_frame_count = 0
        self.max_register_frames = 20
        self.username = ""

    def initUI(self):
        self.setWindowTitle("人脸识别系统")
        self.setMinimumSize(900, 500)
        self.setStyleSheet("background-color: #f0f0f0;")

        self.label = QLabel("请选择功能：")
        self.label.setFont(QFont("Arial", 14))

        self.recognize_btn = QPushButton("识别")
        self.camera_recognize_btn = QPushButton("摄像头识别")
        self.register_btn = QPushButton("注册用户")
        self.batch_register_btn = QPushButton("批量注册")
        self.delete_user_btn = QPushButton("删除用户")
        self.clear_btn = QPushButton("清空显示")
        self.show_users_btn = QPushButton("显示用户信息")

        for btn in [self.recognize_btn, self.camera_recognize_btn, self.register_btn,
                    self.batch_register_btn, self.delete_user_btn, self.clear_btn, self.show_users_btn]:
            btn.setMinimumHeight(40)
            btn.setFont(QFont("Arial", 12))

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("输入用户名")
        self.name_input.setFont(QFont("Arial", 12))
        self.name_input.setMinimumHeight(35)

        self.image_label = QLabel("图像显示区域")
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid gray; background-color: white;")

        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Arial", 14))
        self.result_label.setStyleSheet("color: blue;")
        self.result_label.setWordWrap(True)
        self.result_label.setFixedWidth(220)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.label)
        button_layout.addWidget(self.recognize_btn)
        button_layout.addWidget(self.camera_recognize_btn)
        button_layout.addWidget(self.name_input)
        button_layout.addWidget(self.register_btn)
        button_layout.addWidget(self.batch_register_btn)
        button_layout.addWidget(self.delete_user_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.show_users_btn)
        button_layout.addWidget(self.result_label)
        button_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        main_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.image_label)

        self.setLayout(main_layout)

        self.recognize_btn.clicked.connect(self.recognize_face)
        self.register_btn.clicked.connect(self.register_user)
        self.batch_register_btn.clicked.connect(self.batch_register)
        self.delete_user_btn.clicked.connect(self.delete_user)
        self.camera_recognize_btn.clicked.connect(self.start_camera_recognize)
        self.clear_btn.clicked.connect(self.clear_display)
        self.show_users_btn.clicked.connect(self.show_user_info)

    def clear_display(self):
        self.image_label.clear()
        self.image_label.setText("图像显示区域")
        self.result_label.setText("")

    def recognize_face(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择要识别的图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                QMessageBox.critical(self, "错误", "无法读取图片")
                return

            cropper = FaceCropper("weights/yolov8n-face-lindevs.pt")
            face_crops, box_coords = cropper.crop_faces_with_coords(img)

            for (x1, y1, x2, y2) in box_coords:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            if len(face_crops) == 0:
                self.result_label.setText("未检测到人脸")
                return

            temp_face_path = "temp/croped_face.jpg"
            cv2.imwrite(temp_face_path, face_crops[0])
            user, score = self.recognizer.recognize(temp_face_path)
            os.remove(temp_face_path)

            self.result_label.setText(f"识别结果：\n用户: {user}\n余弦相似度: {score:.4f}")

    def register_user(self):
        self.username = self.name_input.text().strip()
        if not self.username:
            QMessageBox.warning(self, "错误", "请输入用户名")
            return

        save_dir = os.path.join("temp", self.username)
        os.makedirs(save_dir, exist_ok=True)
        self.register_frame_count = 0
        self.register_capture_active = True

        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头")
            return

        QMessageBox.information(self, "提示", "注册模式已启动，请按 X 键拍摄，拍够20张自动完成注册。")
        self.timer_camera.start(30)

    def batch_register(self):
        folder = QFileDialog.getExistingDirectory(self, "选择用户人脸数据目录")
        if folder:
            MODEL_PATH = "weights/arcface_model.pth"
            fr = FaceRegister(MODEL_PATH)
            FaceToolApi.register_auto(fr, folder)
            QMessageBox.information(self, "完成", f"已批量注册用户数据：{folder}")

    def delete_user(self):
        username = self.name_input.text().strip()
        if not username:
            QMessageBox.warning(self, "错误", "请输入要删除的用户名")
            return

        feature_file = os.path.join("features", f"{username}.pt")
        if os.path.exists(feature_file):
            try:
                os.remove(feature_file)
                QMessageBox.information(self, "完成", f"已删除特征文件：{feature_file}")
            except Exception as e:
                QMessageBox.critical(self, "错误", str(e))
        else:
            QMessageBox.warning(self, "未找到", f"未找到特征文件：{feature_file}")

    def start_camera_recognize(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头")
            return
        self.result_label.setText("按S键拍照识别，按Q退出摄像头。")
        self.timer_camera.start(30)

    def display_camera_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            if self.cap and self.cap.isOpened() and self.current_frame is not None:
                self.timer_camera.stop()
                self.cap.release()
                img = self.current_frame.copy()
                cropper = FaceCropper("weights/yolov8n-face-lindevs.pt")
                face_crops, box_coords = cropper.crop_faces_with_coords(img)

                if not face_crops:
                    self.result_label.setText("摄像头画面未检测到人脸")
                    return

                for (x1, y1, x2, y2) in box_coords:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

                temp_face_path = "temp/croped_face.jpg"
                os.makedirs("temp", exist_ok=True)
                cv2.imwrite(temp_face_path, face_crops[0])

                user, score = self.recognizer.recognize(temp_face_path)
                os.remove(temp_face_path)

                self.result_label.setText(f"识别结果：\n用户: {user}\n置信度: {score:.4f}")

        elif event.key() == Qt.Key_X:
            if self.register_capture_active and self.cap and self.cap.isOpened() and self.current_frame is not None:
                save_path = os.path.join("temp", self.username, f"{self.register_frame_count}.jpg")
                cv2.imwrite(save_path, self.current_frame)
                self.register_frame_count += 1
                remaining = self.max_register_frames - self.register_frame_count
                self.result_label.setText(f"注册中，请按X拍照，还需 {remaining} 张")

                if self.register_frame_count >= self.max_register_frames:
                    self.timer_camera.stop()
                    self.cap.release()
                    self.register_capture_active = False

                    cropper = FaceCropper("weights/yolov8n-face-lindevs.pt")
                    raw_data_root = "temp/"
                    dataset_root = "UsersFaceData/"
                    cropper.crop_faces(raw_data_root, dataset_root)

                    shutil.rmtree("temp")
                    os.makedirs("temp")

                    MODEL_PATH = "weights/arcface_model.pth"
                    fr = FaceRegister(MODEL_PATH)
                    FaceToolApi.register_auto(fr, os.path.join("UsersFaceData", self.username))

                    self.image_label.setText("已完成注册")
                    self.result_label.setText("")

        elif event.key() == Qt.Key_Q:
            if self.cap and self.cap.isOpened():
                self.timer_camera.stop()
                self.cap.release()
            self.image_label.setText("摄像头已关闭")
            self.result_label.setText("")

    def show_user_info(self):
        features_dir = "features"
        if not os.path.exists(features_dir):
            QMessageBox.information(self, "提示", "未找到用户数据目录。")
            return

        files = [f for f in os.listdir(features_dir) if f.endswith(".pt")]
        if not files:
            QMessageBox.information(self, "提示", "当前没有注册用户。")
            return

        usernames = [os.path.splitext(f)[0] for f in files]
        user_count = len(usernames)
        user_list = "\n".join(usernames)

        QMessageBox.information(self, "用户信息", f"当前用户数：{user_count}\n用户名列表：\n{user_list}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceApp()
    window.show()
    sys.exit(app.exec_())
