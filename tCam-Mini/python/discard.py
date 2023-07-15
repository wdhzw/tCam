import os
import shutil

def discard_images(scenario, label, discard_indices):
    FOLDER_PATH = "python/dataset"  # Change according to your folder structure
    img_folder = os.path.join(FOLDER_PATH, scenario, label)
    
    # discard the specified images and txt files
    for idx in discard_indices:
        if idx == -1:
            break
        img_file = os.path.join(img_folder, f'{label}_{idx:04}.tiff')
        txt_file = os.path.join(img_folder, f'{label}_{idx:04}.txt')
        if os.path.exists(img_file):
            os.remove(img_file)
        if os.path.exists(txt_file):
            os.remove(txt_file)

    # get a list of remaining files
    remaining_files = sorted(os.listdir(img_folder))

    # prepare to renumber the remaining files
    tiff_files = sorted([file for file in remaining_files if file.endswith(".tiff")])
    txt_files = sorted([file for file in remaining_files if file.endswith(".txt")])

    # renumber the remaining tiff and txt files
    for i in range(len(tiff_files)):
        old_tiff_path = os.path.join(img_folder, tiff_files[i])
        new_tiff_path = os.path.join(img_folder, f'{label}_{i:04}.tiff')
        old_txt_path = os.path.join(img_folder, txt_files[i])
        new_txt_path = os.path.join(img_folder, f'{label}_{i:04}.txt')
        
        shutil.move(old_tiff_path, new_tiff_path)
        shutil.move(old_txt_path, new_txt_path)


if __name__ == "__main__":
    scenario = input("Enter the scenario(office/bedroom/livingroom): ")
    label = input("Enter the label (Human/NonHuman): ")
    indices = input("Enter the indices of images to be discarded, separated by space: ")
    
    # convert string indices to integer
    discard_indices = list(map(int, indices.split()))

    discard_images(scenario, label, discard_indices)
