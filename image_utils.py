from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
import numpy as np

"""
函数说明：
1.load_color_database(file_path)：加载指定位置的数据集并返回一个dataframe；
2.get_base_coefficient(rgb_value):计算原有基础系数；
3.get_scene_coefficient(target_val, db_val):计算RGB增减系数；
4.find_nearest_color(target_rgb, color_database):在颜色数据集中找到指定RGB颜色的相近色号；
"""

def load_color_database(file_path):
    """加载颜色数据，返回颜色数据框"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"颜色数据库文件 {file_path} 不存在")
    
    color_data = pd.read_csv(file_path)
    color_data = color_data.dropna(subset=['R', 'G', 'B', 'num'])
    color_data['R'] = pd.to_numeric(color_data['R'], errors='coerce').fillna(0).astype(int)
    color_data['G'] = pd.to_numeric(color_data['G'], errors='coerce').fillna(0).astype(int)
    color_data['B'] = pd.to_numeric(color_data['B'], errors='coerce').fillna(0).astype(int)
    color_data['R'] = color_data['R'].clip(0, 255) #clip(0,255)表示数值约束再0-255之间
    color_data['G'] = color_data['G'].clip(0, 255)
    color_data['B'] = color_data['B'].clip(0, 255)
    return color_data

def get_base_coefficient(rgb_value):
    """原有基础系数计算"""
    if rgb_value < 0:
        return 0.2
    elif rgb_value > 50:
        return 1.0
    else:
        return 0.3 + (rgb_value / 50) * 0.5

def get_scene_coefficient(target_val, db_val):
    """RGB增减相关系数"""
    if target_val < 80 and db_val < target_val:
        return 0.7
    elif target_val > 170 and db_val > target_val:
        return 0.7
    else:
        return 1.0

def find_nearest_color(target_rgb, color_database):
    """找到与目标RGB值最接近的颜色编号"""
    tr, tg, tb = target_rgb
    color_data = color_database.copy()
    
    r_base = get_base_coefficient(tr)
    g_base = get_base_coefficient(tg)
    b_base = get_base_coefficient(tb)
    
    color_data['r_coeff'] = color_data['R'].apply(lambda x: r_base * get_scene_coefficient(tr, x))
    color_data['g_coeff'] = color_data['G'].apply(lambda x: g_base * get_scene_coefficient(tg, x))
    color_data['b_coeff'] = color_data['B'].apply(lambda x: b_base * get_scene_coefficient(tb, x))
    
    color_data['distance'] = np.sqrt(
        (color_data['r_coeff'] * (color_data['R'] - tr))**2 + 
        (color_data['g_coeff'] * (color_data['G'] - tg))**2 + 
        (color_data['b_coeff'] * (color_data['B'] - tb))**2
    )
    
    nearest_color = color_data.loc[color_data['distance'].idxmin()]
    return nearest_color['num'], (nearest_color['R'], nearest_color['G'], nearest_color['B'])

def process_image_with_color_code(input_path, output_path, color_db_path, scale_factor=0.03, pixel_scale=50):
    """改进的图片处理函数"""
    color_database = load_color_database(color_db_path)
    
    try:
        img = Image.open(input_path).convert('RGB') #转换成RGB模式
    except Exception as e:
        raise Exception(f"打开图片失败: {e}")
    
    #输出与原图大小相同的图纸
    pixel_scale = 1/scale_factor

    original_width, original_height = img.size
    target_width = max(1, int(original_width * scale_factor))
    target_height = max(1, int(original_height * scale_factor))
    #Image.NEAREST：最近邻插值方法
    img_resized = img.resize((target_width, target_height), Image.NEAREST)

    output_width = original_width
    output_height = original_height
    output_img = Image.new("RGB", (output_width, output_height), color="white")
    draw = ImageDraw.Draw(output_img)   

    # 字体加载（兼容不同系统）
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", pixel_scale // 2)
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", pixel_scale // 2)
        except IOError:
            font = ImageFont.load_default()

    # 处理每个像素
    for y in range(target_height):
        for x in range(target_width):
            pixel_r, pixel_g, pixel_b = img_resized.getpixel((x, y))
            color_code, nearest_rgb = find_nearest_color((pixel_r, pixel_g, pixel_b), color_database)
            nr, ng, nb = nearest_rgb
            
            # 绘制像素块
            draw.rectangle(
                [x * pixel_scale, y * pixel_scale,
                 (x + 1) * pixel_scale - 1, (y + 1) * pixel_scale - 1],
                fill=(nr, ng, nb)
            )

            # 绘制颜色编号
            text_color = (255 - nr, 255 - ng, 255 - nb)
            font_size = min(pixel_scale // 2, 12)
            
            try:
                current_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                current_font = ImageFont.load_default()
            
            draw.text(
                (x * pixel_scale + pixel_scale // 2, y * pixel_scale + pixel_scale // 2),
                str(color_code),
                fill=text_color,
                font=current_font,
                anchor="mm"
            )
    
    # 保存结果
    output_img.save(output_path)
    return output_path
