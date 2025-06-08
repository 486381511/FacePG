import os
from pypinyin import lazy_pinyin


# 批量注册
def _register_from_folder(fr, user_name: str, folder: str):
    """[私有] 注册用户人脸，读取文件夹内所有图片（内部使用）"""
    img_paths = [
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    fr.register_user(user_name, img_paths)


def register_auto(fr, folder: str):
    """根据文件夹名自动推断用户名，并注册"""
    folder_name = os.path.basename(folder.rstrip("/\\"))
    user_name = "".join(lazy_pinyin(folder_name))
    _register_from_folder(fr, user_name, folder)


# def main():
#     MODEL_PATH = "weights/arcface_model.pth"
#     fr = FaceRegister(MODEL_PATH)
#     register_auto(fr, r"TrainDatasets\yangmi")
#     register_auto(fr, r"TrainDatasets\yuwenle")