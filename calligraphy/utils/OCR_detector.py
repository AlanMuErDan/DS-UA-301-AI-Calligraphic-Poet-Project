import os
import easyocr
import re
from tqdm import tqdm

def ocr_single_character(image_folder):
    """
    Process all images in a folder using EasyOCR, and return a list of recognized Chinese characters.
    
    Args:
        image_folder (str): Path to the folder containing image files.
        
    Returns:
        list: A list of recognized characters (or None if not a valid single Chinese character).
    """

    reader = easyocr.Reader(['ch_sim'], gpu=True)  # initialize OCR Reader
    
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    
    results = []
    for img_file in tqdm(image_files):
        img_path = os.path.join(image_folder, img_file)
        recognized_text = reader.readtext(img_path, detail=0)  
        if recognized_text:
            text = recognized_text[0].strip()  # take the first recognized string
            if len(text) == 1 and re.match(r'[\u4e00-\u9fff]', text):  # if not Chinese character
                results.append(text)
            else:
                results.append(None)  
        else:
            results.append(None) 
    
    return results

if __name__ == "__main__":
    # Example usage
    # ---------------------------------------
    # Specify your example folder here:
    # ---------------------------------------
    input_folder = "./example_folder"
    recognized_characters = ocr_single_character(input_folder)
    
    print("OCR Results:")
    for i, char in enumerate(recognized_characters):
        print(f"Image {i + 1}: {char}")
    
    print(recognized_characters)