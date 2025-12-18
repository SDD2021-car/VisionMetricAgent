import os
from PIL import Image

# 设置文件夹路径
folder_path = '/data/yjy_data/B2DiffRL_SAR2Opt/controlnet_pretrained/output_image_resize'  # 替换为你的文件夹路径

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.jpeg'):  # 检查是否为JPG或JPEG文件
        # 构建文件路径
        img_path = os.path.join(folder_path, filename)
        os.remove(img_path)
        #
        # # 打开JPG文件
        # img = Image.open(img_path)
        #
        # # 获取文件名（不包括扩展名）
        # name_without_extension = os.path.splitext(filename)[0]
        #
        # # 构建PNG文件保存路径
        # png_path = os.path.join(folder_path, f"{name_without_extension}.png")
        #
        # # 保存为PNG格式
        # img.save(png_path, 'PNG')
        #
        # print(f"已将 {filename} 转换为 {png_path}")
