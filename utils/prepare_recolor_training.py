import os
import shutil
import glob
import random
from PIL import Image
import cv2
import numpy as np

ref_dir = "F:/ImageSet/ColorFix/New folder"
input_dir = "F:/ImageSet/ColorFix/raw"
output_dir = "F:/ImageSet/ColorFix/processed"

supported_image_types = ['.jpg','.jpeg','.png','.webp']

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get reference files
files = glob.glob(f"{ref_dir}/**", recursive=True)
ref_files = [f for f in files if os.path.splitext(f)[-1].lower() in supported_image_types]

# Suffixes
g_suffix = "_G"
f_suffix = "_F"
m_suffix = "_M"
r_suffix = "_R"

# Get input image files
files = glob.glob(f"{input_dir}/**", recursive=True)
image_files = [f for f in files if os.path.splitext(f)[-1].lower() in supported_image_types]

def create_random_mask(image_shape):
    """Create a random mask with basic shapes"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    height, width = image_shape[:2]
    
    # Choose a random shape type
    shape_type = random.choice(['rectangle', 'circle', 'polygon'])
    
    if shape_type == 'rectangle':
        # Random rectangle
        x1 = random.randint(0, width//4)
        y1 = random.randint(0, height//4)
        x2 = random.randint(width*3//4, width)
        y2 = random.randint(height*3//4, height)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
    elif shape_type == 'circle':
        # Random circle
        center_x = random.randint(width//4, width*3//4)
        center_y = random.randint(height//4, height*3//4)
        radius = random.randint(min(width, height)//4, min(width, height)//2)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
    elif shape_type == 'polygon':
        # Random polygon (triangle or quadrilateral)
        num_points = random.choice([3, 4])
        points = []
        for _ in range(num_points):
            x = random.randint(0, width)
            y = random.randint(0, height)
            points.append((x, y))
        pts = np.array([points], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
    
    # Convert single channel mask to 3 channels if needed
    if len(image_shape) == 3:
        mask = cv2.merge([mask, mask, mask])
    
    return mask

for ref_file in ref_files:
    basename = os.path.basename(ref_file)
    filename, ext = os.path.splitext(basename)
    base_filename = filename.replace(r_suffix, "")
    print(f"Processing: {base_filename}")
    
    # Copy reference file to output dir
    shutil.copy(ref_file, f"{output_dir}/{base_filename}{r_suffix}{ext}")
    
    # Process corresponding input files
    for supported_ext in supported_image_types:
        input_path = f"{input_dir}/{base_filename}{supported_ext}"
        if os.path.exists(input_path):
            # Read image with PIL
            image = Image.open(input_path)
            
            # Save original image with G suffix
            image.save(f"{output_dir}/{base_filename}{g_suffix}{supported_ext}")
            
            # Convert to greyscale and save with F suffix
            grey_image = image.convert('L')
            grey_image.save(f"{output_dir}/{base_filename}{f_suffix}{supported_ext}")
            
            # Create random mask image using OpenCV
            img_cv = cv2.imread(input_path)
            if img_cv is not None:
                # Create random mask
                random_mask = create_random_mask(img_cv.shape)
                
                # Determine output extension (use same as input)
                output_ext = supported_ext
                
                # Save mask image with M suffix
                mask_output_path = f"{output_dir}/{base_filename}{m_suffix}{output_ext}"
                cv2.imwrite(mask_output_path, random_mask)
                
                print(f"Created random mask: {mask_output_path}")
            else:
                print(f"Could not read image: {input_path}")

print("Processing complete!")