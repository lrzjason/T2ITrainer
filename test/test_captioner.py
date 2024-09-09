import glob
from captioner.florenceLargeFt import FlorenceLargeFtModelWrapper
from tqdm import tqdm
import os
import cv2
import re

def handle_replace(result):
    result = re.sub(r'A cartoon[a-zA-Z ]*?of ', '', result)
    result = re.sub(r'An animated[a-zA-Z ]*?of ', '', result)
    return result

if __name__ == "__main__":
    # image_path = "F:/ImageSet/sd3_test/1_creative_photo/ComfyUI_temp_zpsmu_00236_.png"
    # image = Image.open(image_path)
    
    # input_dir = "E:/Development/Bilibili-Image-Grapple/classification/output/bomiao"
    input_dir = "F:/ImageSet/pony_caption_output_test"
    # output_dir = "E:/Development/Bilibili-Image-Grapple/classification/output/bomiao_crop_watermark"
    # os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(f"{input_dir}/**", recursive=True)
    image_exts = [".png",".jpg",".jpeg",".webp"]
    image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts]
    # image_files = ["E:/Development/Bilibili-Image-Grapple/classification/output/maileji - Copy/maileji_3.png"]
    # print(image_files)
    model = FlorenceLargeFtModelWrapper()
    # input_dir = "F:/ImageSet/niji"
    # loop input_dir for each image
    for image_file in tqdm(image_files):
        text_file = os.path.splitext(image_file)[0] + ".txt"
        # image_path = os.path.join(input_dir, image_file)
        
        image = cv2.imread(image_file)
        # get webp params
        # filesize = os.path.getsize(image_file) 
        # # print('File: ' + file + ' Size: ' + str(filesize) + ' bytes')
        # filesize_mb = filesize / 1024 / 1024
        # # skip low filesize images
        # if filesize_mb < 0.5:
        #     print("skip low filesize image: ", image_file)
        #     continue
        # lossless, quality = get_webp_params(filesize_mb)
        
        # image = cv2.resize(image, (int(image.shape[1]*0.7), int(image.shape[0]*0.7)), interpolation=cv2.INTER_AREA)
        # ori_image = image.copy()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.open(image_path).convert('RGB')
        result = model.execute(image)
        print("\n")
        print(result)
        result = handle_replace(result)
        print(result)
        
        
        # # read text file
        # with open(text_file, "r", encoding="utf-8") as f:
        #     text = f.read()
        #     new_content = "二次元动漫风格, anime artwork, " + result + ", " + text
        #     # rename original text file to _ori.txt
        #     old_text_file = text_file.replace(".txt","_ori.txt")
        #     if os.path.exists(old_text_file):
        #         continue
        #     # save new content to text file
        #     with open(old_text_file, "w", encoding="utf-8") as ori_f:
        #         ori_f.write(text)
        #         print("save ori content to text file: ", old_text_file)
        #     # save new content to text file
        #     with open(text_file, "w", encoding="utf-8") as new_f:
        #         new_f.write(new_content)
        #         print("save new content to text file: ", text_file)
            
        
        # ############# OCR for watermark ################
        # break
        # result = model.execute(image,other_prompt="<OCR_WITH_REGION>")
        
        # crop_image = False
        # quad_boxes = result["quad_boxes"]
        # for i, quad_box in enumerate(quad_boxes):
        #     x1,y1,x2,y2,x3,y3,x4,y4 = quad_box
            
        #     # only handle fixed bottom region
        #     if y1 > 0.8*image.shape[0]:
        #         cv2.line(image, (0, int(y1)), (image.shape[1], int(y1)), (0, 0, 255), 1)
        #         # crop image
        #         crop_img = image[0:int(y1), :]
        #         crop_image = True
        # # show image for debug
        # # cv2.imshow('Image', image)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        
        # # save cropped image to output_dir
        # file_name, file_ext = os.path.splitext(os.path.basename(image_file))
        # output_path = os.path.join(output_dir, f"{file_name}.webp")
        
        # print("save image: ", output_path)
        # if crop_image:
        #     cv2.imwrite(output_path, crop_img, [int(cv2.IMWRITE_WEBP_QUALITY), quality])
        # else:
        #     cv2.imwrite(output_path, image, [int(cv2.IMWRITE_WEBP_QUALITY), quality])
        # # break