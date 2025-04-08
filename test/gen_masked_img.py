from nudenet import NudeDetector
import cv2
import numpy as np
import glob
import os

def apply_mask(detector, input_dir, output_dir, mask_intensity=30):
    
    
    supported_image_types = ['.jpg','.jpeg','.png','.webp']
    files = glob.glob(f"{input_dir}/**", recursive=True)
    image_files = [f for f in files if os.path.splitext(f)[-1].lower() in supported_image_types]

    gt_suffix = "_G"
    f_suffix = "_F"
    m_suffix = "_M"
    for image_path in image_files:
        # 检测敏感区域
        detections = detector.detect(image_path)
        # 读取图片
        img = cv2.imread(image_path)
        ori_img = img.copy()
        
        basename = os.path.basename(image_path)
        filename, ext = os.path.splitext(basename)
        
        is_sextual = False
        # zero numbpy like img
        mask = np.zeros_like(img)
        
        # 对每个检测区域加马赛克
        for region in detections:
            if region['class'] in ['FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED']:
                is_sextual = True
                x1, y1, width, height = map(int, region['box'])
                # color the area in random
                            # 生成随机颜色 (BGR格式)
                random_color = tuple(np.random.randint(0, 256, 3).tolist())
                
                # 用随机颜色完全覆盖目标区域
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x1+width, y1+height),
                    color=random_color,
                    thickness=-1  # thickness=-1 表示填充整个矩形
                )
                # set mask to 1
                mask[y1:y1+height, x1:x1+width] = 255
        
        if is_sextual:
            # save ori
            cv2.imwrite(f"{output_dir}/{filename}{gt_suffix}{ext}", ori_img)
            # save masked
            cv2.imwrite(f"{output_dir}/{filename}{f_suffix}{ext}", img)
            # save mask
            cv2.imwrite(f"{output_dir}/{filename}{m_suffix}{ext}", mask)
        # break

def main():
    
    # 初始化检测器
    detector = NudeDetector()
    input_dir = "F:\Fapello.Downloader\output\sasha-grey_crop"
    output_dir = "F:\Fapello.Downloader\output\sasha-grey_prepared"
    
    os.makedirs(output_dir, exist_ok=True)
    # 使用示例
    apply_mask(detector,input_dir, output_dir)

if __name__ == "__main__":
    main()