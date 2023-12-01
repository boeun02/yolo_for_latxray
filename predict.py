from ultralytics import YOLO

model = YOLO("yolov8m-seg-custom.pt")

model.predict(source="6.png", 
              show=False,
              save=True, 
              show_labels=True, 
              show_conf=True, 
              conf=0.5, 
              save_txt=False, 
              save_crop=False, 
              line_width=2,
              box=False, 
              visualize=True)
