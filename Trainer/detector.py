import os
# import glob
# from IPython.display import Image, display

HOME = os.getcwd()
print(HOME)

# Install yolo
# from IPython import display
# display.clear_output()

import ultralytics
ultralytics.checks()
from ultralytics import YOLO

# from IPython.display import display, Image

# Download dataset
# try: 
#     os.mkdir("datasets") 
# except OSError as error: 
#     print(error) 
# os.chdir("datasets")

# from roboflow import Roboflow
# rf = Roboflow(api_key="Hd8Xn8GCPjFv0Ebfk5Ry")
# project = rf.workspace("fire-extinguisher").project("fireextinguisher-z5atr")
# dataset = project.version(2).download("yolov8")

# os.chdir("..")
# print(os.getcwd())


#### TRAIN ####
model = YOLO("runs/detect/train/weights/best.pt")
model.train(task="detect", epochs=100, batch=16, imgsz=800, data=f"{HOME}/FireExtinguisher-2/data.yaml", single_cls=True, image_weights=True, plots=True)
#### TRAIN ####

# #### IMAGES ####
# # Confusion matrix
# Image(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=600)

# # Results
# Image(filename=f'{HOME}/runs/detect/train/results.png', width=600)

# # Predictions
# Image(filename=f'{HOME}/runs/detect/train/val_batch0_pred.jpg', width=600)
# #### IMAGES ####


#### VALIDATE ####
# model.val(task="detect", model="runs/detect/train/weights/best.pt", data=f"{HOME}/FireExtinguisher-2/data.yaml")
#### VALIDATE ####

#### INFERENCE ####
results = model.predict(task="detect", model=f"{HOME}runs/detect/train/weights/best.pt", conf=0.25, source="tests", save=True, save_txt=True, augment=True)

boxes = results[0].boxes
box = boxes[0]  # returns one box
print("--------------------")
print(box.xyxy.tolist())
print("--------------------")
print(box.xywh.tolist())
print("--------------------")
print(box.xyxyn.tolist())
print("--------------------")
print(box.xywhn.tolist())
print("--------------------")
print(box.conf.tolist())
print("--------------------")
print(box.data.tolist())

coords = ["Left", "Top", "Right", "Bottom"]

for result in results:
    boxes = result.boxes
    for box in boxes:
        for i in range(len(coords)):
            print(f"{coords[i]}: {box.xyxy.tolist()[0][i]}")


# for image_path in glob.glob(f'{HOME}/runs/detect/predict/*.jpeg')[:]:
#       display(Image(filename=image_path, width=600))
#       print("\n")
#### INFERENCE ####