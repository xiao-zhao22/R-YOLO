import warnings
warnings.filterwarnings('ignore')
import cv2, os, shutil
import numpy as np
from ultralytics import YOLO

# def get_video_cfg(path):
#     video = cv2.VideoCapture(path)
#     size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#     fps = int(video.get(cv2.CAP_PROP_FPS))
#     return cv2.VideoWriter_fourcc(*'XVID'),size,fps

def plot_and_counting(result):
    image_plot = result.plot()
    box_count = result.boxes.shape[0]
    cv2.putText(image_plot,f'{box_count}rebars',(2000, 280), cv2.FONT_HERSHEY_SIMPLEX, 3,(255, 0, 0), 8)
    return image_plot

if __name__ == '__main__':
    output_dir = 'result'
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    os.makedirs(output_dir,exist_ok=True)

    model = YOLO(r'C:\Users\86183\Desktop\yolov8\ultralytics-main\runs\train\yolov8s\weights\best.pt')#  权重

    #------------for images-----
    for result in model.predict(source=r'C:\Users\86183\Desktop\yolov8\ultralytics-main\img\1.jpg',
                                stream=True,
                                imgsz=640,
                                save=False,
                                conf=0.5,
                                iou=0.4
                                ):


        image_plot =plot_and_counting(result)
        #image_plot =
        cv2.imwrite(f'{output_dir}/{os.path.basename(result.path)}',image_plot)
