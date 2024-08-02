import os
import sys
sys.path.append('F:/T2ITrainer/aesthetic')
import aesthetic_predict


def main():
    input_dir = "F:/ImageSet/vit_train/crop_predict_cropped/highest"
    # eval
    ae_model,image_encoder,preprocess,device = aesthetic_predict.init_model()

    total_score = 0
    subsets = os.listdir(input_dir)
    for subset in subsets:
        subset_dir = os.path.join(input_dir, subset)
        subset_score = 0
        files = os.listdir(subset_dir)
        for image_file in files:
            if not image_file.endswith('.webp'):
                continue
            
            image_path = os.path.join(subset_dir,image_file)
            subset_score += aesthetic_predict.predict(ae_model,image_encoder,preprocess,image_path,device)
        
        # get average score
        subset_score /= len(files)
        total_score+=subset_score
        print(f"{subset}:{subset_score}")
    
    total_score /= len(subsets)
    print(f"total_score:{total_score}")

if __name__ == "__main__":
    main()