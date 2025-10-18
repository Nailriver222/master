import os
import random
import numpy as np
from PIL import Image



# ========== ユーザーが指定する部分 ==========
# 入力画像（グレースケール）のファイル名
image_files = [
    'img\\left_w.bmp', 
    'img\\center_w.bmp', 
    'img\\right_w.bmp'
]

# 出力フォルダを指定
output_folder = 'C:\\Users\\nailr\\VScode Projects\\master\\output\\output_rgb'
# ============================================

def load_grayscale_image(path):
    return Image.open(path).convert('L')

def create_random_color_image(size):
    color = tuple(random.randint(0, 255) for _ in range(3))
    return Image.new('RGB', size, color), color

def multiply_images(gray_img, color_img):
    gray_np = np.array(gray_img) / 255.0
    color_np = np.array(color_img)
    result_np = (color_np * gray_np[..., None]).astype(np.uint8)
    return Image.fromarray(result_np)

def add_all_images_hdr(image_list):
    result = np.zeros_like(np.array(image_list[0]), dtype=np.float32)
    for img in image_list:
        result += np.array(img, dtype=np.float32)
    max_val = np.max(result)
    if max_val > 0:
        result = (result / max_val) * 255
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)

def process_images(image_paths, output_dir):
    ensure_output_dir(output_dir)
    colored_images = []

    for i, path in enumerate(image_paths):
        gray_img = load_grayscale_image(path)
        color_img, color = create_random_color_image(gray_img.size)

        print(f"Processing: {path} with color {color}")
        colored_img = multiply_images(gray_img, color_img)
        colored_images.append(colored_img)

        # 個別画像の保存
        filename = f"colored_image_{i+1}.png"
        colored_img.save(os.path.join(output_dir, filename))

    # 最終合成
    final_image = add_all_images_hdr(colored_images)
    final_path = os.path.join(output_dir, 'final_combined_image.png')
    final_image.save(final_path)

    print(f"✅ 全処理完了。画像は '{os.path.abspath(output_dir)}' に保存されました。")
    return final_image, colored_images

# ========== 実行 ==========
if __name__ == '__main__':
    final_result, individual_results = process_images(image_files, output_folder)
