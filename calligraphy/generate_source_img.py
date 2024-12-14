from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm

# -------------------------------
# define your output folder here
# -------------------------------
output_folder = "./output"
os.makedirs(output_folder, exist_ok=True)  

# load Mac font 
# font_path = "/System/Library/Fonts/STHeiti Light.ttc"  # 宋体
font_path = "/System/Library/Fonts/PingFang.ttc" # 平方
font_size = 100 
font = ImageFont.truetype(font_path, font_size)

characters = [chr(i) for i in range(0x4e00, 0x9fff)]

for char in tqdm(characters):
    img = Image.new("L", (128, 128), "white")
    draw = ImageDraw.Draw(img)
    text_bbox = draw.textbbox((0, 0), char, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_position = ((128 - text_width) // 2, (128 - text_height) // 2-8)
    draw.text(text_position, char, font=font, fill="black")
    img.save(os.path.join(output_folder, f"{char}.png"))

print("Finished!!")