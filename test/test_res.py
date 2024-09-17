import pandas as pd

RESOLUTION_CONFIG = {
    1024: [
        # extra resolution for testing
        (1536, 1536),
        (1344, 1344),
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

def get_buckets(resolution=1024):
    resolution_set = RESOLUTION_CONFIG[resolution]
    horizontal_resolution_set = resolution_set
    vertical_resolution_set = [(height,width) for width,height in resolution_set]
    all_resolution_set = horizontal_resolution_set + vertical_resolution_set
    all_resolution_set = pd.Series(all_resolution_set).drop_duplicates().tolist()
    buckets = {}
    for resolution in all_resolution_set:
        buckets[f'{resolution[0]}x{resolution[1]}'] = []
    return buckets

get_buckets()