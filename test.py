import os

output_dir = "F:/ImageSet/comat_kolors_512_2"
metadata = {
    'images':[]
}

prompt_file = "F:/CoMat/collected_data/abc5k_2.txt"
with open(prompt_file, 'r', encoding="utf-8") as f:
    prompts = f.readlines()
    for i,prompt in enumerate(prompts):
        index = i+2500
        text_file = os.path.join(output_dir, f"{index}.txt")
        npz_path = text_file.replace(".txt",".npkolors")
        if os.path.exists(text_file):
            metadata["images"].append({
                "prompt":prompt,
                # 'npz_path_md5':get_md5_by_path(npz_path),
                "npz_path":npz_path,
                "txt_path":text_file
            })
            continue
        else:
            print('text_file not exist, ', text_file)
            
print(metadata)