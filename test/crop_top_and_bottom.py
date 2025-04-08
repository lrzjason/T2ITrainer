
import cv2
import numpy as np
import glob
import os

def crop_top_and_bottom(image_path, crop_margin):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    crop_top = int(height * crop_margin)
    crop_bottom = int(height * crop_margin)
    cropped_image = image[crop_top:-crop_bottom, :]
    return cropped_image

def main():
    input_dir = "F:\Fapello.Downloader\output\sasha-grey"
    output_dir = "F:\Fapello.Downloader\output\sasha-grey_crop"
    
    os.makedirs(output_dir, exist_ok=True)
    
    supported_image_types = ['.jpg','.jpeg','.png','.webp']
    files = glob.glob(f"{input_dir}/**", recursive=True)
    image_files = [f for f in files if os.path.splitext(f)[-1].lower() in supported_image_types]

    crop_margin = 0.08
    
    for image_path in image_files:
        # crop image top and bottom in crop_margin
        image = crop_top_and_bottom(image_path, crop_margin)
        cv2.imwrite(f"{output_dir}/{os.path.basename(image_path)}", image)
    
if __name__ == "__main__":
    main()