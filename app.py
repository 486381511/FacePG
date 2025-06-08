from register import FaceRegister
from recognize import FaceRecognizer
from yoloFaceCutFace import FaceCropper
import cv2
import os
import FaceToolApi

def main():
    # 确保临时文件夹存在
    os.makedirs("temp", exist_ok=True)

    img_folder = "testImg"  # 这里是文件夹

    # 人脸裁剪器，使用 YOLOv8n-face 模型进行人脸检测
    yoloface_weights = "weights/yolov8n-face-lindevs.pt"  # 人脸检测模型权重
    cropper = FaceCropper(yoloface_weights)

    # 注册用户，注册成功后会在 features 目录下生成用户特征文件，下次识别时会自动加载这些特征文件。不用重新注册
    MODEL_PATH = "weights/arcface_model.pth"
    fr = FaceRegister(MODEL_PATH)

    # 批量注册示例（可根据需要打开）
    FaceToolApi.register_auto(fr, "UsersFaceData/dengziqi")
    FaceToolApi.register_auto(fr, "UsersFaceData/huge")
    FaceToolApi.register_auto(fr, "UsersFaceData/lym")
    

    recognizer = FaceRecognizer(MODEL_PATH, threshold=0.30)

    for filename in os.listdir(img_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(img_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue

        face_crops, box_coords = cropper.crop_faces_with_coords(img)
        if not face_crops:
            print(f"{filename} 未检测到人脸")
            continue

        cv2.imshow("Detected Face", face_crops[0])
        cv2.waitKey(0)

        temp_face_path = "temp/croped_face.jpg"
        cv2.imwrite(temp_face_path, face_crops[0])  # 保存裁剪后的人脸图像

        user, score = recognizer.recognize(temp_face_path)
        print(f"{filename} 识别到：{user}，分数：{score:.4f}")

        if os.path.exists(temp_face_path):
            os.remove(temp_face_path)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
