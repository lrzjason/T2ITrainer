from PIL import Image
import requests
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
import cv2
import os
from tqdm import tqdm 
import numpy



BASE_RESOLUTION = 1024

RESOLUTION_SET = [
    (1024, 1024),
    (1152, 896),
    (1216, 832),
    (1344, 768),
    (1536, 640),
]


# return closest_ratio and width,height closest_resolution
def get_nearest_resolution(image):
    height, width, _ = image.shape
    
    # get ratio
    image_ratio = width / height

    horizontal_resolution_set = RESOLUTION_SET
    horizontal_ratio = [round(width/height, 2) for width,height in RESOLUTION_SET]

    vertical_resolution_set = [(height,width) for width,height in RESOLUTION_SET]
    vertical_ratio = [round(height/width, 2) for height,width in vertical_resolution_set]

    target_ratio = horizontal_ratio
    target_set = horizontal_resolution_set
    if width<height:
        target_ratio = vertical_ratio
        target_set = vertical_resolution_set

    # Find the closest vertical ratio
    closest_ratio = min(target_ratio, key=lambda x: abs(x - image_ratio))
    closest_resolution = target_set[target_ratio.index(closest_ratio)]

    return closest_ratio,closest_resolution

def simple_scale(image,scale_with_height,closest_resolution):
    height, width, _ = image.shape
    # print("ori size:",width,height)

    if scale_with_height:  # 横が長い→縦を合わせる
        scale = closest_resolution[1] / height
    else:
        scale = closest_resolution[0] / width
    
    resized_size = (int(width * scale + 0.5), int(height * scale + 0.5))
    print(f"ori ratio:{width}/{height}")
    print(f"closest ratio:{resized_size[0]}/{resized_size[1]}")
    # resize image to target resolution
    return cv2.resize(image, resized_size)

def save_webp(image,filename,suffix="",output_dir=""):
    if suffix != '':
        filename = f"{filename}_{suffix}.webp"

    if output_dir != "":
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir,filename)
        
    # cv2.imshow("resized_image", resized_image)
    cv2.imwrite(filename, image, [int(cv2.IMWRITE_WEBP_QUALITY), 95])

def main():
    input_dir = "F:/ImageSet/vit_train/crop_predict_cropped/original"
    output_dir = "F:/ImageSet/vit_train/crop_predict_cropped/scaled_original"
    # create output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in tqdm(os.listdir(input_dir)):
        # Check if the file is an image by its extension
        if file.endswith((".webp")) or file.endswith((".jpg")) or file.endswith((".png")):
            file_path = os.path.join(input_dir,file)
            filename, ext = os.path.splitext(os.path.basename(file_path))

            
            # read image
            image = Image.open(file_path)
            

            # pil_image = Image.open(image_path).convert('RGB')
            open_cv_image = numpy.array(image)
            # # Convert RGB to BGR
            image = open_cv_image[:, :, ::-1].copy()


            # get image width, height
            height, width, _ = image.shape

            # get nearest resolution
            closest_ratio,closest_resolution = get_nearest_resolution(image)

            # we need to expand the closest resolution to target resolution before cropping
            scale_ratio = closest_resolution[0] / closest_resolution[1]
            image_ratio = width / height

            scale_with_height = True
            # referenced kohya ss code
            if image_ratio < scale_ratio: 
                scale_with_height = False

            simple_image = simple_scale(image,scale_with_height,closest_resolution)
            save_webp(simple_image,file,output_dir=output_dir)


if __name__ == "__main__":
    main()