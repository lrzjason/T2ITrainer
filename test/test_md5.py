from hashlib import md5

def get_md5_by_path(file_path):
    try:
        with open(file_path, 'rb') as f:
            return md5(f.read()).hexdigest()
    except:
        print(f"Error getting md5 for {file_path}")
        return ''

# a = "123"
# b = "123"
a_path = "F:/ImageSet/recolor_part/test/bestvisionhdr12k_000006_F.txt"
b_path = "F:/ImageSet/recolor_part/test/bestvisionhdr12k_000007_F.txt"
a_md5 = get_md5_by_path(a_path)
b_md5 = get_md5_by_path(b_path)

if a_md5 == b_md5:
    print("is the same")
else:
    print("not the same")