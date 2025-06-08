import os
import torch
from torchvision import transforms
from PIL import Image
from train import ResNetEmbedding  # 你的模型定义文件

class FaceRegister:
    """
    人脸注册类

    接口说明：
    - __init__(model_path: str, feature_dir: str = "features")
        初始化注册器，加载模型，指定保存特征的目录。

    - register_user(user_name: str, img_paths: list[str]) -> None
        给定用户名和该用户多张人脸图片路径，提取embedding后取均值，
        并保存特征到本地文件（feature_dir/user_name.pt）。
    """

    def __init__(self, model_path: str, feature_dir: str = "features"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNetEmbedding()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["backbone_state"])
        self.model.to(self.device)
        self.model.eval()

        self.feature_dir = feature_dir
        os.makedirs(self.feature_dir, exist_ok=True)

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

    def register_user(self, user_name: str, img_paths: list[str]) -> None:
        embeddings = []
        for path in img_paths:
            emb = self._extract_embedding(path)
            embeddings.append(emb)

        mean_embedding = torch.stack(embeddings).mean(dim=0)
        save_path = os.path.join(self.feature_dir, f"{user_name}.pt")
        torch.save(mean_embedding, save_path)
        print(f"用户 '{user_name}' 注册完成，特征保存到：{save_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python register.py 用户名 img1.jpg img2.jpg ...")
        sys.exit(1)

    username = sys.argv[1]
    images = sys.argv[2:]

    # 请替换为你实际模型文件路径
    MODEL_PATH = "weights/arcface_model.pth"
    fr = FaceRegister(MODEL_PATH)
    fr.register_user(username, images)
