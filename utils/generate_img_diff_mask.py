import cv2
import numpy as np
import os
import glob

def smooth_mask(binary_mask, blur_size=5, epsilon_ratio=0.005):
    """
    高级平滑处理函数
    参数：
    blur_size: 模糊核大小（必须为奇数）
    epsilon_ratio: 轮廓近似精度比率（值越小轮廓保留越精细）
    """
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(binary_mask, (blur_size, blur_size), 0)
    
    # 重新阈值处理
    _, smoothed = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    
    # 轮廓近似优化
    contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_contours = []
    
    for cnt in contours:
        # 计算轮廓周长用于近似精度
        epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx_contours.append(approx)
    
    # 绘制平滑后的轮廓
    smooth_mask = np.zeros_like(binary_mask)
    cv2.drawContours(smooth_mask, approx_contours, -1, 255, -1)
    
    return smooth_mask

def generate_masks(input_dir, output_dir, threshold=30, morph_ops=True):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的文件类型
    supported_image_types = ['.jpg','.jpeg','.png','.webp']
    
    # 递归获取文件
    files = glob.glob(os.path.join(input_dir, '**'), recursive=True)
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in supported_image_types and "_F" in f]

    # 形态学参数配置
    morph_params = {
        'close_kernel': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)),  # 椭圆核更符合自然形状
        'expand_kernel': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)),
        'smooth_kernel': (9,9)  # 高斯模糊核大小（必须奇数）
    }

    for file in image_files:
        try:
            # 构建G文件路径
            basename = os.path.basename(file)
            filename, ext = os.path.splitext(basename)
            
            # get index of factual suffix
            g_basename = filename[:filename.index('_F')] +'_G' + ext
            
            g_path = os.path.join(os.path.dirname(file), g_basename)
            if not os.path.exists(g_path):
                print(f"警告：未找到对应G文件 {g_path}")
                continue

            # 读取图像
            img_f = cv2.imread(file)
            img_g = cv2.imread(g_path)
            if img_f is None or img_g is None:
                print(f"图像读取失败：{file}")
                continue

            # 计算差异
            diff = cv2.absdiff(img_f, img_g)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            _, binary_mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)

            
            if morph_ops:
                # 三阶段形态学优化
                # 1. 初步闭运算连接区域
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, 
                                             morph_params['close_kernel'], iterations=2)
                
                # 2. 膨胀-填洞-腐蚀流程
                binary_mask = cv2.dilate(binary_mask, morph_params['expand_kernel'], iterations=2)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE,
                                             morph_params['close_kernel'], iterations=3)
                binary_mask = cv2.erode(binary_mask, morph_params['expand_kernel'], iterations=2)
                
                # 3. 高级平滑处理
                binary_mask = smooth_mask(binary_mask, 
                                        blur_size=morph_params['smooth_kernel'][0],
                                        epsilon_ratio=0.003)

            # 保存结果
            basename = os.path.basename(file)
            filename, ext = os.path.splitext(basename)
            output_path = os.path.join(output_dir, f"{filename}_M{ext}")
            cv2.imwrite(output_path, binary_mask)
            print(f"生成：{output_path}")

        except Exception as e:
            print(f"处理 {file} 时出错：{str(e)}")

if __name__ == "__main__":
    input_dir = "F:/ImageSet/Anime/cloth_removal"
    output_dir = "F:/ImageSet/Anime/cloth_removal"
    
    generate_masks(
        input_dir=input_dir,
        output_dir=output_dir,
        threshold=20,
        morph_ops=True
    )