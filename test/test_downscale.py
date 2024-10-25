import cv2

RESOLUTION_CONFIG = {
    512: [
        # extra resolution for testing
        # (1536, 1536),
        # (672, 672),
        (512, 512),
        (768, 512), # 1.5
        (576, 448), # 1.5
        (608, 416), # 1.5
    ],
    1024: [
        # extra resolution for testing
        # (1536, 1536),
        # (1344, 1344),
        (1024, 1024),
        (1344, 1024),
        (1152, 896), # 1.2857
        (1216, 832), # 1.46
        (1344, 768), # 1.75
        (1536, 640), # 2.4
    ],
    2048: [
        (2048, 2048),
        (2304, 1792), # 1.2857
        (2432, 1664), # 1.46
        (2688, 1536), # 1.75
        (3072, 1280), # 2.4
    ]
}

def closest_mod_64(value):
    return value - (value % 64)
# return closest_ratio and width,height closest_resolution
def get_nearest_resolution(image, resolution=1024):
    height, width, _ = image.shape
    if height==width and (width <= 1344 and height <= 1344):
        closest_pixel = closest_mod_64(width)
        return 1, (closest_pixel,closest_pixel)
    
    resolution_set = RESOLUTION_CONFIG[resolution]
    
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

def crop_image(image,resolution=1024):
    height, width, _ = image.shape
    # get nearest resolution
    closest_ratio,closest_resolution = get_nearest_resolution(image,resolution=resolution)
    # we need to expand the closest resolution to target resolution before cropping
    scale_ratio = closest_resolution[0] / closest_resolution[1]
    image_ratio = width / height
    scale_with_height = True
    # referenced kohya ss code
    if image_ratio < scale_ratio: 
        scale_with_height = False
    try:
        image,crop_x,crop_y = simple_center_crop(image,scale_with_height,closest_resolution)
    except Exception as e:
        print(e)
        raise e
    
    return image,crop_x,crop_y

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

    crop_x = 0
    crop_y = 0
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

    # print(f"ori ratio:{width/height}")
    height, width, _ = cropped_image.shape  
    # print(f"cropped ratio:{width/height}")
    # print(f"closest ratio:{closest_resolution[0]/closest_resolution[1]}")
    # resize image to target resolution
    # return cv2.resize(cropped_image, closest_resolution)
    return resize(cropped_image,closest_resolution),crop_x,crop_y


interpolation = cv2.INTER_AREA
def resize(img,resolution):
    # return cv2.resize(img,resolution,interpolation=cv2.INTER_AREA)
    return cv2.resize(img,resolution, interpolation=interpolation)
if __name__ == "__main__":
    image_path = "F:/ImageSet/kolors_cosplay/train_backup/gasuto/58929526_p0.jpg"
    image = cv2.imread(image_path)
    cropped_image,_,_ = crop_image(image)
    # print(crop_image)
    # save crop image
    cv2.imwrite("crop_image_INTER_AREA.jpg", cropped_image)
    interpolation = cv2.INTER_CUBIC
    crocropped_imagep_image,_,_ = crop_image(image)
    cv2.imwrite("crop_image_INTER_CUBIC.jpg", cropped_image)