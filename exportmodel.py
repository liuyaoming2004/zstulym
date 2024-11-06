from ultralytics import YOLOv10

model = YOLOv10("yolov10n.pt")
model = YOLOv10(r'D:\桌面\yolov10-main\2022337621219best.pt')

model.export(format='onnx')