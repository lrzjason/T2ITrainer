


BASE_RESOLUTION = 1024

RESOLUTION_SET = [
    (1024, 1024),
    (1152, 896),
    (1216, 832),
    (1344, 768),
    (1536, 640),
]

def get_buckets():
    horizontal_resolution_set = RESOLUTION_SET
    vertical_resolution_set = [(height,width) for width,height in RESOLUTION_SET]
    all_resolution_set = horizontal_resolution_set + vertical_resolution_set[1:]
    buckets = {}
    for resolution in all_resolution_set:
        buckets[f'{resolution[0]}x{resolution[1]}'] = []
    return buckets

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