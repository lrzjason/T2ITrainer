from PIL import Image
import os
import glob

def check_image_pairs(directory="."):
    """
    Check that for each image base name, the _R, _T, _G variants have the same size.
    Raises ValueError with filenames if mismatch found.
    """
    # Get all image files (supports common image extensions)
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.gif')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))

    # Group by base name (remove _R, _T, _G suffix)
    groups = {}
    suffixes = ['_R', '_T', '_G']

    for filepath in image_files:
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)

        for suf in suffixes:
            if name.endswith(suf):
                base_name = name[:-len(suf)]
                if base_name not in groups:
                    groups[base_name] = {}
                groups[base_name][suf] = filepath
                break

    # Check each group
    for base_name, files in groups.items():
        # Ensure all three suffixes are present
        missing = [suf for suf in suffixes if suf not in files]
        if missing:
            print(f"⚠️  Warning: Group '{base_name}' is missing: {missing}. Skipping.")
            continue

        sizes = {}
        for suf in suffixes:
            try:
                with Image.open(files[suf]) as img:
                    sizes[suf] = img.size  # (width, height)
            except Exception as e:
                raise RuntimeError(f"Error opening {files[suf]}: {e}")

        # Check if all sizes match
        first_size = sizes[suffixes[0]]
        mismatches = [suf for suf in suffixes if sizes[suf] != first_size]

        if mismatches:
            mismatch_names = [files[suf] for suf in mismatches]
            mismatch_names.append(files[suffixes[0]])  # Include reference too for context
            raise ValueError(
                f"Size mismatch in group '{base_name}': {mismatch_names} — "
                f"Sizes: {[f'{suf}:{sizes[suf]}' for suf in suffixes]}"
            )

    print("✅ All image groups have matching sizes.")

# Example usage
if __name__ == "__main__":
    try:
        check_image_pairs(r"F:\ImageSet\yanse")  # Change path as needed
    except Exception as e:
        print(f"❌ Error: {e}")