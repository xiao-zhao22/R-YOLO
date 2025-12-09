from ultralytics import YOLO
if __name__ == '__main__':
  pth_path = r"C:\Users\86183\Desktop\yolov8\ultralytics-main\runs\detect\yolov8s\weights\best.pt"
  test_path = r"C:\Users\86183\Desktop\yolov8\ultralytics-main\datasets\images\test1"
  # Load a model
  # model = YOLO('yolov8n.pt')  # load an official model
  model = YOLO(pth_path)  # load a custom model
  # Predict with the model  ,show_labels= False,show_conf= False,show_boxes= False
  results = model(test_path, save=True, conf=0.5,iou= 0.4)  # predict on an image