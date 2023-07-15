import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

def display_images(scenario, label, start, end):
    FOLDER_PATH = "python/dataset"  # Change according to your folder structure
    img_folder = os.path.join(FOLDER_PATH, scenario, label)

    # create a list of filenames based on the range
    filenames = [f'{label}_{i:04}.tiff' for i in range(start, end+1)]
    # Skip 'partofbody' images and non-existent files
    filenames = [filename for filename in filenames if not ("partofbody" in filename or not os.path.exists(os.path.join(img_folder, filename)))]
    # calculate the layout for subplot
    n = len(filenames)
    if n == 0:
        print('Index out of range!')
        return
    ncols = 3  # change to fit your needs
    nrows = n // ncols + bool(n % ncols)

    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))  # adjust the figure size
    
    for i, filename in enumerate(filenames):
        img_path = os.path.join(img_folder, filename)
        img = Image.open(img_path)
        ax[i // ncols, i % ncols].imshow(img, cmap='hot') 
        ax[i // ncols, i % ncols].set_title(filename)
    
    # hide the axes of extra subplots
    for j in range(i+1, nrows*ncols):
        fig.delaxes(ax.flatten()[j])
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    scenario = input("Enter the scenario(office/bedroom/livingroom): ")
    label = input("Enter the label (Human/NonHuman): ")
    start = int(input("Enter the start index (inclusive): "))
    end = int(input("Enter the end index (inclusive): "))
    display_images(scenario, label, start, end)
