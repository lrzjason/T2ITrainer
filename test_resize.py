import numpy as np
from utils.image_utils_kolors import get_nearest_resolution, simple_center_crop
from PIL import Image
import cv2

if __name__ == "__main__":
    image_path = "F:/ImageSet/kolors_cosplay/train/maileji/maileji_2.webp"
    
    
    
    # open_cv_image = numpy.array(image)
    # # Convert RGB to BGR
    # image = open_cv_image[:, :, ::-1].copy()
    
    try:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is not None:
            # Convert to RGB format (assuming the original image is in BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            print(f"Failed to open {image_path}.")
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")
    
    # set meta data
    height, width, _ = image.shape
    # get nearest resolution
    closest_ratio,closest_resolution = get_nearest_resolution(image,resolution=1024)
    # print('init closest_resolution',closest_resolution)

    # we need to expand the closest resolution to target resolution before cropping
    scale_ratio = closest_resolution[0] / closest_resolution[1]
    image_ratio = width / height

    scale_with_height = True
    # referenced kohya ss code
    if image_ratio < scale_ratio: 
        scale_with_height = False
    try:
        # image = simple_center_crop(image,scale_with_height,closest_resolution)
        image,crop_x,crop_y = simple_center_crop(image,scale_with_height,closest_resolution)
        # save_webp(simple_crop_image,filename,'simple',os.path.join(output_dir,"simple"))
    except Exception as e:
        print(e)
        raise e
    # set meta data
    image_height, image_width, _ = image.shape
    print(image_height,image_width)
    image = Image.fromarray(image)
    image.show()
    