import base64
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tcam import TCam
import cv2

COUNTDOWN_TIMER = 3
TCAM_IP = "192.168.1.120"
TCAM_PORT = 5001
FOLDER_PATH = "python/dataset"

# Prompt user for scenario
scenario = input("Please enter the scenario (office, bedroom, livingroom): ")

# Prompt user for label
label = input("Please enter the label (Human, NonHuman): ")
if label not in ['Human', 'NonHuman']:
    print("Invalid label, abort")
    exit()

LABEL_PATH = "./"+FOLDER_PATH + "/" + scenario + "/" + label
label_name = "/" + label.lower() + "_"
snap_interval = 2

path_used = LABEL_PATH 
path_file_name = label_name

print("Saving to ", path_used)
files = []
for (dirpath, dirname, filename) in os.walk(path_used):
    files.extend(filename)
    break

saved_count = int(len(files)/2)

camera = TCam(responseTimeout=15)

try:
    camera.connect(TCAM_IP)
except:
    print("tCam not found, abort")
    exit()

user_input = input("How many images to gather? Enter in integer > ")
try:
    image_count = int(user_input)
except:
    print("Invalid input, abort")
    exit()

while COUNTDOWN_TIMER > 0:
    print("Timer: {}".format(COUNTDOWN_TIMER))
    time.sleep(1)
    COUNTDOWN_TIMER -= 1

for save in range(saved_count, image_count + saved_count):
    time.sleep(snap_interval)
    uint8_img_string = base64.b64decode(camera.get_image()["radiometric"])
    image_np_arr = np.frombuffer(uint8_img_string, dtype=np.uint16)
    image_np_arr = image_np_arr.reshape((120, 160))
    cv2.imwrite(path_used+ path_file_name+"{:04d}.tiff".format(save), image_np_arr)
    file = open(path_used + path_file_name+"{:04d}.txt".format(save), "wb")
    file.write(image_np_arr)
    print("saving image: {:04d}".format(save))
    file.close()

    # New: Plot the image
    plt.imshow(image_np_arr, cmap='gist_heat', interpolation='nearest')
    plt.show(block=False)
    plt.pause(1) # pauses for 1 second
    plt.close()

camera.shutdown()

exit(0)
