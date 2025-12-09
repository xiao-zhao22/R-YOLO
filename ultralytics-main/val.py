import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# BILIBILI UP 魔傀面具
# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO

# 精度小数点保留位数修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第五点

if __name__ == '__main__':
    model = YOLO(r"C:\Users\86183\Desktop\yolov8\ultralytics-main\runs\train\yolov8s\weights\best.pt")
    model.val(data=r"C:/Users/86183/Desktop/yolov8/ultralytics-main/data.yaml",
              split='test',
              imgsz=640,
              batch=4,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/test',
              name='exp',
              )