from ultralytics import YOLO
import torch
from openvino.inference_engine import IECore
model = YOLO('yolov10n.pt')

if __name__ == '__main__':
    model.train(data='yolo-bvn.yaml',workers=1,epochs=10,batch=16)


#yolo detect train data=/root/autodl-tmp/ultralytics-main11/datasets/bvn/yolo-bun.yaml model=yolo11n.pt epochs=200 batch=16 imgsz=640 device=0 optimizer="Adam"

#yolov10-main/yolo-bun.yaml

#yolo detect train data=/root/autodl-tmp/yolov10-main1/yolov10-main/yolo-bun.yaml model=yolov10n.pt epochs=150 batch=16 imgsz=640 device=0
