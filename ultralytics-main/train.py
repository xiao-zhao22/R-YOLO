from ultralytics import YOLO

if __name__ == '__main__':
    model_yaml = r"C:\Users\86183\Desktop\yolov8\ultralytics-main\yolov8s.yaml"
    data_yaml = r"C:\Users\86183\Desktop\yolov8\ultralytics-main\data.yaml"
    pre_model = r"C:\Users\86183\Desktop\yolov8\ultralytics-main\yolov8s.pt"
    model = YOLO(model_yaml, task='detect').load(pre_model)
    # build from YAML and transfer weights
    # Train the model
    results = model.train(data=data_yaml, epochs=100, imgsz=640, batch=4, workers=0)