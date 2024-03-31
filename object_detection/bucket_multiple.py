from PIL import Image
import requests
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
import cv2
import os
from tqdm import tqdm 
import numpy

import sys
sys.path.append('F:/T2ITrainer/aesthetic')
import aesthetic_predict


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

def save_webp(image,filename,suffix="",output_dir=""):
    if suffix != '':
        filename = f"{filename}_{suffix}.webp"

    if output_dir != "":
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir,filename)
        
    # cv2.imshow("resized_image", resized_image)
    cv2.imwrite(filename, image, [int(cv2.IMWRITE_WEBP_QUALITY), 95])

def get_biggest_features_bbox(model,processor,image,draw=False):
    height, width, _ = image.shape
    prompt = "<grounding> the main objects of this image are:"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

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
        # print(entity)
        name,_,bboxes = entity
        for bbox in bboxes:
            x, y, x2, y2 = bbox
            if draw:
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

    buffer_ratio = 32 / BASE_RESOLUTION
    x_buffer = int(width * buffer_ratio+0.5)
    if x_buffer < 32:
        x_buffer = 32
    y_buffer = int(height * buffer_ratio+0.5)
    if y_buffer < 32:
        y_buffer = 32
    min_x = min_x if min_x - x_buffer < 0 else min_x - x_buffer
    min_y = min_y if min_y - y_buffer < 0 else min_y - y_buffer
    max_x = width if max_x + x_buffer > width else max_x + x_buffer
    max_y = height if max_y + y_buffer > height else max_y + y_buffer

    if draw:
        image = cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 3)

    return min_x, min_y, max_x, max_y

def simple_center_crop(image,scale_with_height,closest_resolution):
    height, width, _ = image.shape
    # print("ori size:",width,height)
    if scale_with_height: 
        up_scale = height / closest_resolution[1]
    else:
        up_scale = width / closest_resolution[0]

    expanded_closest_size = (int(closest_resolution[0] * up_scale + 0.5), int(closest_resolution[1] * up_scale + 0.5))
    
    diff_x = abs(expanded_closest_size[0] - width)
    diff_y = abs(expanded_closest_size[1] - height)

    # crop extra part of the resized images
    if diff_x>0:
        crop_x =  diff_x //2
        cropped_image = image[:,  crop_x:width-diff_x+crop_x]
    elif diff_y>0:
        crop_y =  diff_y//2
        cropped_image = image[crop_y:height-diff_y+crop_y, :]
    else:
        # 1:1 ratio
        cropped_image = image

    print(f"ori ratio:{width/height}")
    height, width, _ = cropped_image.shape  
    print(f"cropped ratio:{width/height}")
    print(f"closest ratio:{closest_resolution[0]/closest_resolution[1]}")
    # resize image to target resolution
    return cv2.resize(cropped_image, closest_resolution)

def pixel_preserved_crop(image,scale_with_height,closest_resolution,model,processor):
    height, width, _ = image.shape
    # print("ori size:",width,height)
    if scale_with_height: 
        up_scale = height / closest_resolution[1]
    else:
        up_scale = width / closest_resolution[0]

    expanded_closest_size = (int(closest_resolution[0] * up_scale + 0.5), int(closest_resolution[1] * up_scale + 0.5))
    
    diff_x = abs(expanded_closest_size[0] - width)
    diff_y = abs(expanded_closest_size[1] - height)

    min_x,min_y,max_x,max_y = get_biggest_features_bbox(model,processor,image)

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
    cropped_image = image[crop_y:crop_y2, crop_x:crop_x2]

    
    # resize image to target resolution
    return cv2.resize(cropped_image, closest_resolution)

def features_centered_crop(image,scale_with_height,closest_resolution,model,processor,draw=False):
    height, width, _ = image.shape

    min_x,min_y,max_x,max_y = get_biggest_features_bbox(model,processor,image)

    ori_center = (int(width / 2), int(height / 2))
    new_center = (int((max_x + min_x) / 2), int((max_y + min_y) / 2))

    # calculate new_center different from ori_center
    diff_x = abs(ori_center[0] - new_center[0])
    diff_y = abs(ori_center[1] - new_center[1])

    # handle features using full width and height
    try:
        left_ratio = min_x/(width-max_x+min_x)
    except:
        left_ratio = 1

    try:
        upper_ratio = min_y/(height-max_y+min_y)
    except:
        upper_ratio = 1

    left_crop = int(left_ratio*diff_x)
    upper_crop = int(upper_ratio*diff_y)

    crop_x = left_crop
    crop_x2 = width-diff_x+left_crop
    crop_y = upper_crop
    crop_y2 = height-diff_y+upper_crop
    # crop the image
    # for the feature center
    cropped_image = image[crop_y:crop_y2, crop_x:crop_x2]
    # print(cropped_image.shape[1],cropped_image.shape[0])

    area_points = {
        'upper_area': ((0, 0), (width, min_y-8)),
        'left_area': ((0, 0), (min_x-8, height)),
        'right_area': ((max_x+8, 0), (width, height)),
        'bottom_area': ((0, max_y+8), (width, height))
    }

    # compare areas to get the biggest area
    # Calculate the areas
    upper_area = width * (min_y - 8)
    left_area = (min_x - 8) * height
    right_area = (width - (max_x + 8)) * height
    bottom_area = width * (height - (max_y + 8))

    # Create a dictionary to hold the area values
    areas = {
        'upper_area': upper_area,
        'left_area': left_area,
        'right_area': right_area,
        'bottom_area': bottom_area
    }


    # Find the largest area
    largest_area = max(areas, key=areas.get)
    # print(f"The largest area is {largest_area} with an area of {areas[largest_area]}")

    # draw the largest_area
    if draw:
        cv2.imshow("largest_features", image)
        image = cv2.rectangle(image, area_points[largest_area][0], area_points[largest_area][1], (0, 0, 255), 3)
    
    
    image = cropped_image
    height, width, _ = cropped_image.shape
    
    # # get ratio
    image_ratio = width / height
    closest_ratio,closest_resolution = get_nearest_resolution(cropped_image)
    
    # print('cropped closest_resolution',closest_resolution)

    # =========================================================
    # Scale image to closest resolution
    # Referenced from Kohya ss train_util.py
    # =========================================================

    ar_reso = closest_resolution[0] / closest_resolution[1]
    if image_ratio > ar_reso:  # 横が長い→縦を合わせる
        scale = closest_resolution[1] / height
    else:
        scale = closest_resolution[0] / width
    
    resized_size = (int(width * scale + 0.5), int(height * scale + 0.5))
    resized_image = cv2.resize(cropped_image, resized_size)
    
    height, width, _ = resized_image.shape
    # print("cropped extra image",width, height)
    
    # crop extra part of the resized images
    if resized_size[0] > closest_resolution[0]:
        diff_x = abs((resized_size[0] - closest_resolution[0]))
        crop_x =  diff_x //2
        resized_image = resized_image[:,  crop_x:width-diff_x+crop_x]
    elif resized_size[1] > closest_resolution[1]:
        diff_y = abs((resized_size[1] - closest_resolution[1]))
        crop_y =  diff_y//2
        resized_image = resized_image[crop_y:height-diff_y+crop_y, :]

    # resize image to target resolution
    return resized_image

def apply_crop(model,processor,image_path,output_dir,highest_count,ae_model,image_encoder,preprocess,device):
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename, ext = os.path.splitext(os.path.basename(image_path))

    # pil_image = Image.open(image_path).convert('RGB')
    # open_cv_image = numpy.array(pil_image)
    # # Convert RGB to BGR
    # image = open_cv_image[:, :, ::-1].copy()

    # read image
    image = Image.open(image_path)
    image_ori = image.copy()

    # pil_image = Image.open(image_path).convert('RGB')
    open_cv_image = numpy.array(image)
    # # Convert RGB to BGR
    image = open_cv_image[:, :, ::-1].copy()

    # image = cv2.imread(image_path)  # Replace with your image path
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        raise Exception(f"Error: Could not read image from {image_path}")
    height, width, _ = image.shape

    # get nearest resolution
    closest_ratio,closest_resolution = get_nearest_resolution(image)
    # print('init closest_resolution',closest_resolution)

    # we need to expand the closest resolution to target resolution before cropping
    scale_ratio = closest_resolution[0] / closest_resolution[1]
    image_ratio = width / height

    scale_with_height = True
    # referenced kohya ss code
    if image_ratio < scale_ratio: 
        scale_with_height = False

    # print(resized_image.shape[1],resized_image.shape[0])
    try:
        simple_crop_image = simple_center_crop(image,scale_with_height,closest_resolution)
        save_webp(simple_crop_image,filename,'simple',os.path.join(output_dir,"simple"))
    except Exception as e:
        print(e)
        raise e
    
    
    try:
        pixel_preserved_crop_image = pixel_preserved_crop(image,scale_with_height,closest_resolution,model,processor)
        save_webp(pixel_preserved_crop_image,filename,'preserved',os.path.join(output_dir,"preserved"))
    except Exception as e:
        print(e)
        raise e

    try:
        features_centered_crop_image = features_centered_crop(image,scale_with_height,closest_resolution,model,processor)
        save_webp(features_centered_crop_image,filename,'centered',os.path.join(output_dir,"centered"))
    except Exception as e:
        print(e)
        raise e



    simple_score = aesthetic_predict.predict(ae_model,image_encoder,preprocess,simple_crop_image,device)
    preserved_score = aesthetic_predict.predict(ae_model,image_encoder,preprocess,pixel_preserved_crop_image,device)
    centered_score = aesthetic_predict.predict(ae_model,image_encoder,preprocess,features_centered_crop_image,device)

    print('filename',filename)
    print('simple_score',simple_score)
    print('preserved_score',preserved_score)
    print('centered_score',centered_score)


    highest_score = simple_score
    highest_suffix = 'simple'
    if preserved_score>highest_score:
        highest_score = preserved_score
        highest_suffix = 'preserved'
        highest_count['preserved'] +=1
        # highest_count['preserved_list'].append(filename)
    elif centered_score>highest_score:
        highest_score = centered_score
        highest_suffix = 'centered'
        highest_count['centered'] +=1
        # highest_count['centered_list'].append(filename)
    else:
        highest_count['simple'] +=1
        # highest_count['simple_list'].append(filename)
    print('highest_count',highest_count)

    highest_dir = os.path.join(output_dir,"highest")
    if not os.path.exists(highest_dir):
        os.makedirs(highest_dir)
    
    save_webp(simple_crop_image,filename,highest_suffix,os.path.join(highest_dir,highest_suffix))

    return simple_score,preserved_score,centered_score,highest_score

def average(lst): 
    return sum(lst) / len(lst) 
def main():
    input_dir = "F:/ImageSet/vit_train/crop_predict"
    output_dir = "F:/ImageSet/vit_train/crop_predict_cropped"

    simple_score_list = []
    preserved_score_list = []
    centered_score_list = []
    highest_score_list = []

    highest_count = {
        'simple':0,
        # 'simple_list':[],
        'preserved':0,
        # 'preserved_list':[],
        'centered':0,
        # 'centered_list':[]
    }

    # eval
    ae_model,image_encoder,preprocess,device = aesthetic_predict.init_model()

    # load kosmos2 model
    model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224").to(device)
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224", return_tensors="pt")

    crop_methods = ["centered","preserved","simple"]

    for file in tqdm(os.listdir(input_dir)):
        # Check if the file is an image by its extension
        if file.endswith((".webp")) or file.endswith((".jpg")) or file.endswith((".png")):
            filename,_ = os.path.splitext(file)
            skip = False
            for crop_method in crop_methods:
                if os.path.exists(os.path.join(output_dir,"highest",crop_method,f"{filename}_{crop_method}.webp")):
                    highest_count[crop_method]+=1
                    skip = True
                    break
            if skip:
                continue

            file_path = os.path.join(input_dir,file)

            try:
                simple_score,preserved_score,centered_score,highest_score = apply_crop(model,
                    processor,file_path,output_dir,
                    highest_count,ae_model,image_encoder,
                    preprocess,device)    
                
                simple_score_list.append(simple_score)
                preserved_score_list.append(preserved_score)
                centered_score_list.append(centered_score)
                highest_score_list.append(highest_score)
            except Exception as e: 
                print(e)
                # print(f'Error: Could not read image from {file_path}')
            

    simple_average = average(simple_score_list)
    preserved_average = average(preserved_score_list)
    centered_average = average(centered_score_list)
    highest_average = average(highest_score_list)

    print(f'len:{len(simple_score_list)} simple_average:{simple_average}')
    print(f'len:{len(preserved_score_list)} preserved_average:{preserved_average}')
    print(f'len:{len(centered_score_list)} centered_average:{centered_average}')
    print(f'len:{len(highest_score_list)} highest_average:{highest_average}')

    print('highest_count',highest_count)
    print('done')

if __name__ == "__main__":
    main()