Environment Configuration：
    Software Usage：Anaconda, Pycharm2025.2.5
    1:opening Anaconda prompt
    Enter the following commands in sequence：
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
        conda config --set show_channel_urls yes

        pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

        conda create -n yolo-common python==3.9

        conda activate yolo-common

        nvidia-smi
        (Note: Check the highest supported CUDA version on your system. This computer uses CUDA version 11.7.)

        conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c nvidia
    2：Open the ultralytics-main folder in the code using PyCharm.

        Select the preconfigured yolo-common environment in the lower-right corner.

        Open the terminal in the lower-left corner.

        Click the dropdown button to open the command prompt. Ensure you are in the yolo-common environment, then enter the following command:

            pip install -e.
    3:Re-enter the Anaconda prompt and navigate to the configured environment.

        conda activate yolo-common

        //Enter the following command

        conda install -c defaults intel-openmp -f
    4：Open PyCharm, open the terminal in the lower-left corner, click the dropdown button to enter the command prompt, ensure you are in the yolo-common environment, and enter the following command:

        pip install einops

        pip install timm
Environment configuration complete.

Experiment Reproduction:

Modify the data.yaml file path:
    path:ultralytics-main>data.yaml
    Change the absolute paths of the following files:
        path: C:\Users\86183\Desktop\yolov8\ultralytics-main\datasets 
        train: C:\Users\86183\Desktop\yolov8\ultralytics-main\datasets\images\train 
        val: C:\Users\86183\Desktop\yolov8\ultralytics-main\datasets\images\val  
        test: C:\Users\86183\Desktop\yolov8\ultralytics-main\datasets\images\val

Ablation experiments(Table3) ：

    YOLOv8s:
        Code changes:
        1：path:ultralytics-main>ultralytics>utils>loss.py(line 227-231)
        Ensure self.use_wiseiou = False
        
        # WiseIOU
                self.use_wiseiou = False
                if self.use_wiseiou:
                    self.wiou_loss = WiseIouLoss(ltype='MPDIoU', monotonous=False, inner_iou=False, focaler_iou=True)
        2：path:ultralytics-main>train.py
        Change the absolute path of the file
        For example:
        model_yaml = r"C:\Users\86183\Desktop\yolov8\ultralytics-main\yolov8s.yaml"
        data_yaml = r"C:\Users\86183\Desktop\yolov8\ultralytics-main\data.yaml"
        pre_model = r"C:\Users\86183\Desktop\yolov8\ultralytics-main\yolov8s.pt"
        
        Click Run to complete training of the original YOLOv8 network.

    Model 1(using FM_WIoU loss):
        1：path:ultralytics-main>ultralytics>utils>loss.py(line 227-231)
        Ensure self.use_wiseiou = True
        
        # WiseIOU
                self.use_wiseiou = True
                if self.use_wiseiou:
                    self.wiou_loss = WiseIouLoss(ltype='MPDIoU', monotonous=False, inner_iou=False, focaler_iou=True)
        2：path:ultralytics-main>train.py
        Click Run to complete training the YOLOv8 network using the FM_WIoU loss.

    Model 2(using detection head):
        1：path:ultralytics-main>ultralytics>utils>loss.py(line 227-231)
        Ensure self.use_wiseiou = False
        
        # WiseIOU
                self.use_wiseiou = False
                if self.use_wiseiou:
                    self.wiou_loss = WiseIouLoss(ltype='MPDIoU', monotonous=False, inner_iou=False, focaler_iou=True)
        2：path:ultralytics-main>train.py
        Modify the following code
        model_yaml = r"C:\Users\86183\Desktop\yolov8\ultralytics-main\yolov8s_P2.yaml"
        Click Run to complete training the YOLOv8 network using the detection head.
    
    Model 3(using FM_WIoU loss and detection head):
        1：path:ultralytics-main>ultralytics>utils>loss.py(line 227-231)
        Ensure self.use_wiseiou = True
        
        # WiseIOU
                self.use_wiseiou = True
                if self.use_wiseiou:
                    self.wiou_loss = WiseIouLoss(ltype='MPDIoU', monotonous=False, inner_iou=False, focaler_iou=True)
        2：path:ultralytics-main>train.py    
       Modify the following code
        model_yaml = r"C:\Users\86183\Desktop\yolov8\ultralytics-main\yolov8s_P2.yaml"
        Click Run to complete training the YOLOv8 network using the FM_WIoU loss and detection head.
    Model 4(using FM_WIoU loss and ASF-YOLO):
        1：path:ultralytics-main>ultralytics>utils>loss.py(line 227-231)
        Ensure self.use_wiseiou = True
        
        # WiseIOU
                self.use_wiseiou = True
                if self.use_wiseiou:
                    self.wiou_loss = WiseIouLoss(ltype='MPDIoU', monotonous=False, inner_iou=False, focaler_iou=True)
        2：path:ultralytics-main>train.py    
        Modify the following code
        model_yaml = r"C:\Users\86183\Desktop\yolov8\ultralytics-main\yolov8s-ASF-P2.yaml"
        Click Run to complete training the YOLOv8 network using the FM_WIoU loss and ASF-YOLO.

Table 4's Experimental Data Using Table 3

The test set experiments(Table5) ：
    path:ultralytics-main>val.py
        Modify the following code
        Change the absolute path of the file
        model = YOLO(r"C:\Users\86183\Desktop\yolov8\ultralytics-main\runs\train\yolov8s\weights\best.pt")
        model.val(data=r"C:/Users/86183/Desktop/yolov8/ultralytics-main/data.yaml",
        
        Experiment using the optimal weights from YOLOV8 or R-YOLO.

Comparison of test set metrics considering size（Table6）：
    1. path:ultralytics-main>data.yaml
        Change the following code paths to test1 (normal group) or test2 (oversize and undersize group).
        
        test: C:\Users\86183\Desktop\yolov8\ultralytics-main\datasets\images\val
        
        test: C:\Users\86183\Desktop\yolov8\ultralytics-main\datasets\images\test1
        
        test: C:\Users\86183\Desktop\yolov8\ultralytics-main\datasets\images\test2
    2.path:ultralytics-main>val.py
        Modify the following code
        model = YOLO(r"C:\Users\86183\Desktop\yolov8\ultralytics-main\runs\train\yolov8s\weights\best.pt")
        Experiment using the optimal weights from YOLOv8 or R-YOLO.

Testing effect diagram(Fig13):
    path:ultralytics-main>detect_counting.py
    Simply adjust the weights and image paths to perform the detection.
    Change the absolute path of the file
        model = YOLO(r'C:\Users\86183\Desktop\yolov8\ultralytics-main\runs\train\yolov8s\weights\best.pt')

        #------------for images-----
        for result in model.predict(source=r'C:\Users\86183\Desktop\yolov8\ultralytics-main\img\1.jpg',

The key data and code paths in the remaining papers are as follows:

    Original photo: YOLOV8>ultralytics-main>img
    
    datasets: YOLOV8>ultralytics-main>datasets
    
    FM_WIoU loss: YOLOV8>ultralytics-main>ultralytics>utils>loss.py(line 215-231)
    
    P2-Additional detection head: YOLOV8>ultralytics-main>yolov8s_P2.yaml
    
    ASF-YOLO: YOLOV8>ultralytics-main>yolov8s-ASF-P2.yaml
    
    train: YOLOV8>ultralytics-main>train.py
    
    test/val: YOLOV8>ultralytics-main>val.py
    
    detect_counting: YOLOV8>ultralytics-main>detect_counting.py
    
    Manuscript_Table 3: YOLOV8>ultralytics-main>runs>train
    
    Manuscript_Fig.11: YOLOV8>ultralytics-main>Fig.11
    
    Manuscript_Table 5+6：YOLOV8>ultralytics-main>Table5+Table6
    
    Manuscript_Fig.13: YOLOV8>ultralytics-main>Fig.13
    
    other methods: YOLOV8>ultralytics-main>other_yaml