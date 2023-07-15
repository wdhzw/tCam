import os

FOLDER_PATH = "python/dataset"  # Change according to your folder structure
scenarios = ["office", "bedroom", "livingroom"]
labels = ["Human", "NonHuman"]

for scenario in scenarios:
    print(f"Scenario: {scenario}")
    for label in labels:
        label_folder = os.path.join(FOLDER_PATH, scenario, label)
        num_images = len(os.listdir(label_folder))
        print(f"   {label}: {num_images/2} images")
    print()
