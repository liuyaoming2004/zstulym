from ultralytics import YOLO
import os

data_path = r'D:\桌面\yolov10-main\yolo-bvn.yaml'  # YAML 文件指向您的数据集
model = YOLO(r'D:\桌面\yolov10-main\zybest.pt')  # 加载您的训练模型

# 继续您的导出代码
model.export(format="openvino", int8=True, data=data_path)
