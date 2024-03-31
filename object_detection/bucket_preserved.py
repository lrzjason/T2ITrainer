from PIL import Image
import requests
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
import cv2
import os

resolution_set = [
    (1024, 1024),
    (1152, 896),
    (1216, 832),
    (1344, 768),
    (1536, 640),
]

# return closest_ratio and width,height closest_resolution
def get_nearest_resolution(image,resolution_set):
    height, width, _ = image.shape
    
    # get ratio
    image_ratio = width / height

    horizontal_resolution_set = resolution_set
    horizontal_ratio = [round(width/height, 2) for width,height in resolution_set]

    vertical_resolution_set = [(height,width) for width,height in resolution_set]
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

def main():
    model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    filename = "x1b413e1dea60b813"
    subdir = "000031"
    image_path = f"F:/ImageSet/improved_aesthetics_6.5plus/output/{subdir}/{filename}.webp"

    filename, ext = os.path.splitext(os.path.basename(image_path))

    image = cv2.imread(image_path)  # Replace with your image path

    # get nearest resolution
    closest_ratio,closest_resolution = get_nearest_resolution(image,resolution_set)
    print('init closest_resolution',closest_resolution)

    height, width, _ = image.shape
    print("ori size:",width,height)

    # we need to expand the closest resolution to target resolution before cropping
    scale_ratio = closest_resolution[0] / closest_resolution[1]
    image_ratio = width / height
    # referenced kohya ss code
    if image_ratio > scale_ratio: 
        up_scale = height / closest_resolution[1]
    else:
        up_scale = width / closest_resolution[0]
    
    expanded_closest_size = (int(closest_resolution[0] * up_scale + 0.5), int(closest_resolution[1] * up_scale + 0.5))
    
    diff_x = abs(expanded_closest_size[0] - width)
    diff_y = abs(expanded_closest_size[1] - height)
    print("up_scale",up_scale)
    print("expanded_closest_size",expanded_closest_size)
    print("diff_x",diff_x)
    print("diff_y",diff_y)

    image_ori = image.copy()

    prompt = "<grounding> the main objects of this image are:"

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=64,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    _, entities = processor.post_process_generation(generated_text)

    min_x = 0
    min_y = 0
    max_x2 = 0
    max_y2 = 0


    # please help to draw the bounding box
    for entity in entities:
        print(entity)
        name,_,bboxes = entity
        for bbox in bboxes:
            x, y, x2, y2 = bbox
            start_pt = (int(x*width+0.5), int(y*height+0.5))
            end_pt = (int(x2*width+0.5), int(y2*height+0.5))
            # print(start_pt, end_pt)
            image = cv2.rectangle(image, start_pt, end_pt, (0, 255, 0), 2)

            if x < min_x or min_x == 0:
                min_x = x
            if y < min_y or min_y == 0:
                min_y = y
            if x2 > max_x2 or max_x2 == 0:
                max_x2 = x2
            if y2 > max_y2 or max_y2 == 0:
                max_y2 = y2

    min_x = int(min_x*width+0.5)
    min_y = int(min_y*height+0.5)

    max_x = int(max_x2*width+0.5)
    max_y = int(max_y2*height+0.5)

    buffer = 32
    min_x = min_x if min_x - buffer < 0 else min_x - buffer
    min_y = min_y if min_y - buffer < 0 else min_y - buffer
    max_x = width if max_x + buffer > width else max_x + buffer
    max_y = height if max_y + buffer > height else max_y + buffer

    image = cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 3)

    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # don't need to cal right and bottom, using width - diff_axios + axios_crop to get the x2

    # handle features using full width and height
    try:
        left_ratio = min_x/(width-max_x+min_x)
    except:
        left_ratio = 1

    try:
        upper_ratio = min_y/(height-max_y+min_y)
    except:
        upper_ratio = 1

    # int would round down but we use width - diff_axios + axios_crop to get the otherside
    left_crop = int(left_ratio*diff_x)
    upper_crop = int(upper_ratio*diff_y)

    crop_x = left_crop
    crop_x2 = width-diff_x+left_crop
    crop_y = upper_crop
    crop_y2 = height-diff_y+upper_crop

    # crop the image
    # for the feature center
    cropped_image = image_ori[crop_y:crop_y2, crop_x:crop_x2]
    # print(cropped_image.shape[1],cropped_image.shape[0])

    # resize image to target resolution
    resized_image = cv2.resize(cropped_image, closest_resolution)
    # print(resized_image.shape[1],resized_image.shape[0])
    # cv2.imshow("resized_image", resized_image)
    cv2.imwrite(f"{filename}_preserved.webp", resized_image, [int(cv2.IMWRITE_WEBP_QUALITY), 95])
    # cv2.waitKey(0)

if __name__ == "__main__":
    main()