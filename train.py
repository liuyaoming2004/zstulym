from ultralytics import YOLO
import torch
import torch.nn as nn
from ultralytics import YOLO

# 这里可以添加您其他的自定义模块
# from your_custom_modules import Conv, C2f, SCDown, Dropout, SPPF, PSA, v10Detect


def main():
    # 加载模型（使用配置文件）
    model = YOLO('yolov10n.pt')

    # 训练模型
    train_results = model.train(
        data=r'D:\桌面\yolov10-main\yolo-bvn.yaml',
        epochs=100,
        imgsz=640,
        device=0,
        batch=8,
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0005,
        project='runs/detect',
        name='1029_10_52',
        close_mosaic=0,
        save=True,
        save_period=-1
    )

    # 评估模型在验证集上的性能
    metrics = model.val(device=0)


if __name__ == '__main__':
    main()
