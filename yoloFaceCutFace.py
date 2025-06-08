import os
import cv2
from ultralytics import YOLO

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import cv2
from ultralytics import YOLO
from pypinyin import lazy_pinyin
from typing import List, Tuple

class FaceCropper:
    """
    人脸裁剪类（基于YOLOFace模型）

    接口说明：
    - __init__(weights_path: str, conf_thresh: float = 0.5)
        初始化裁剪器，加载YOLOFace模型权重，设置置信度阈值。

    - crop_faces(input_root: str, output_root: str) -> None
        批量裁剪人脸。遍历 input_root 目录（结构：中文/图片.jpg），
        裁剪后输出为 output_root/pinyin_name/xxx_face_i.jpg。

    - crop_faces_with_coords(img) -> (List[ndarray], List[List[int]])
        输入一张图像，返回裁剪后人脸图像列表与对应坐标列表。
    """

    def __init__(self, weights_path: str, conf_thresh: float = 0.5):
        self.model = YOLO(weights_path)
        self.conf_thresh = conf_thresh

    def chinese_to_pinyin(self, text: str) -> str:
        # 将中文字符串转换为拼音（无声调，无空格）
        return "".join(lazy_pinyin(text))

    def crop_faces(self, input_root: str, output_root: str) -> None:
        os.makedirs(output_root, exist_ok=True)

        for person_name in os.listdir(input_root):
            person_folder = os.path.join(input_root, person_name)
            if not os.path.isdir(person_folder):
                continue

            pinyin_name = self.chinese_to_pinyin(person_name)
            save_folder = os.path.join(output_root, pinyin_name)
            os.makedirs(save_folder, exist_ok=True)

            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                face_crops, boxes = self.crop_faces_with_coords(img)

                base_name = os.path.splitext(img_name)[0]
                base_name_py = self.chinese_to_pinyin(base_name)

                for i, face_crop in enumerate(face_crops):
                    save_path = os.path.join(save_folder, f"{base_name_py}_face_{i}.jpg")
                    cv2.imwrite(save_path, face_crop)

                print(f"{person_name}/{img_name} → 检测到 {len(boxes)} 张人脸，保存至 {save_folder}")
    
    # 外部调用api
    def crop_faces_with_coords(self, img) -> Tuple[List, List[List[int]]]:
        results = self.model(img)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        face_crops = []
        box_coords = []

        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score < self.conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box)
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            face_crops.append(face_crop)
            box_coords.append([x1, y1, x2, y2])

        return face_crops, box_coords


if __name__ == "__main__":
    raw_data_root = "celebrity_images"  # 原始数据路径，结构：celebrity_images/PersonName/*.jpg
    dataset_root = "TrainDatasets"           # 裁剪后人脸输出目录
    yoloface_weights = "weights/yolov8n-face-lindevs.pt"  # 人脸检测模型权重

    # 创建类，并且加载人脸位置检测权重
    cropper = FaceCropper(yoloface_weights)
    cropper.crop_faces(raw_data_root, dataset_root)
