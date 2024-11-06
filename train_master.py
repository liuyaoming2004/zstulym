from ultralytics import YOLO

def main():
    # 加载模型
    model = YOLO("yolov10n.pt")

    # 训练模型，设置 save=True 和 save_period=-1 来保存最好的模型
    train_results = model.train(
        data=r'D:\桌面\yolov10-main\yolo-bvn.yaml',
        epochs=1,
        imgsz=640,
        device=0,
        batch=8,
        optimizer='Adam',
        lr0=0.0005,
        weight_decay=0.0005,
        project='runs/detect',
        name='111',
        close_mosaic = 0,
        save=True,
        save_period=-1
    )
    # 评估模型在验证集上的性能
    metrics = model.val(device=0)

if __name__ == '__main__':
    main()