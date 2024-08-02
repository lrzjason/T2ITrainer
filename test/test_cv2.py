import cv2
import numpy as np

image_path = "F:/ImageSet/kolors_all_train/anime_character_dataset/alice/0002315_Joël Jurion.jpg"

try:
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is not None:
        # Convert to RGB format (assuming the original image is in BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        print(f"Failed to open {image_path}.")
except Exception as e:
    print(f"An error occurred while processing {image_path}: {e}")
