import os
import torch
from torchvision import transforms
from PIL import Image
from train import ResNetEmbedding
from torch.nn.functional import cosine_similarity

class FaceRecognizer:
    """
    人脸识别类

    在人脸识别中，我们提取出每张脸的 特征向量（embedding），然后用 余弦相似度（cosine similarity） 来比较两张脸的相似程度。

    接口说明：
    - __init__(model_path: str, feature_dir: str = "features", threshold: float = 0.65)
        初始化识别器，加载模型，读取已注册用户特征文件夹，设置相似度阈值。

    - recognize(img_path: str) -> (str, float)
        输入一张人脸图片路径，提取embedding，与数据库中所有用户embedding计算余弦相似度，
        返回识别出的用户名和相似度。如果相似度低于阈值，返回 "Unknown"。
    """

    def __init__(self, model_path: str, feature_dir: str = "features", threshold: float = 0.65):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNetEmbedding()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["backbone_state"])
        self.model.to(self.device)
        self.model.eval()

        self.feature_dir = feature_dir
        self.threshold = threshold

        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def _extract_embedding(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(img).squeeze(0).cpu()
        return embedding

    def recognize(self, img_path: str) -> tuple[str, float]:
        query_embedding = self._extract_embedding(img_path)

        best_score = -1
        best_user = "Unknown"

        for feat_file in os.listdir(self.feature_dir):
            if not feat_file.endswith(".pt"):
                continue
            user_name = os.path.splitext(feat_file)[0]
            db_embedding = torch.load(os.path.join(self.feature_dir, feat_file))
            sim = cosine_similarity(query_embedding, db_embedding, dim=0).item()

            if sim > best_score:
                best_score = sim
                best_user = user_name

        if best_score < self.threshold:
            best_user = "Unknown"

        print(f"识别结果: {best_user}，相似度: {best_score:.4f}")
        return best_user, best_score


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python recognize.py 待识别图片.jpg")
        sys.exit(1)

    MODEL_PATH = "weights/arcface_model.pth"
    recognizer = FaceRecognizer(MODEL_PATH)
    recognizer.recognize(sys.argv[1])
