import skimage.color as skcolor
import numpy as np

# 测试：创建一个 RGB 颜色并转换为 Lab
test_rgb = np.array([[[128, 128, 128]]], dtype=np.uint8) / 255.0
test_lab = skcolor.rgb2lab(test_rgb)

print("skikit-image 安装成功！")
print(f"中灰色的 Lab 值为: {test_lab[0][0]}")