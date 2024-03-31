
from rembg import remove
from PIL import Image
import numpy


# image_path ="F:/ImageSet/openxl2_leftover/1_1_ganyu_fullbody/6003224.jpg"

image_path = "F:/ImageSet/improved_aesthetics_6.5plus/output/000000/xac8cc449c3b0a9f.webp"

image = Image.open(image_path)

output = remove(image)
# Check if the image has an alpha channel
if output.mode == 'RGBA':
    # Split the image into channels
    r, g, b, alpha = output.split()
    # Save the alpha channel as the mask
    alpha.show()