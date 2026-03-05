cls
RD /S /Q "C:\Users\chinm\Desktop\00_FILES\00_ECE4\4415_Capstone_Engineering_Design_Project\Camera\Custom_Dataset\runs"
yolo task=detect mode=train data="data.yaml" model=yolov8n.pt epochs=50 imgsz=640 batch=16 device=0