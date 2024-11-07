# 基于 YOLOv10 的报纸杂志版面分析项目

## 一、项目概述
本项目旨在利用 YOLOv10 目标检测模型对报纸杂志版面进行分析，识别版面中的各类元素，如标题、正文、图片、图表等，实现对版面结构的理解与分析，为后续的内容提取、排版优化等工作提供基础。

## 二、项目流程

### （一）环境准备

1. **安装依赖库**
   - 确保已安装 `ultralytics` 库，用于模型的训练与推理。
   - 安装 `torch` 库，用于深度学习相关计算。
   - 安装 `OpenVINO` 工具包，用于模型的转换与推理加速。

2. **下载 YOLOv10 源码**
   - 从网络上下载 YOLOv10 源码，并将划分好的数据集移动到根目录下。

### （二）训练模型

1. **编写 .yaml 文件**
   - 根据数据集的特点，撰写对应的 .yaml 文件，配置数据集路径、类别等信息。

2. **训练模型（命令行形式）**
   - 使用以下命令行进行训练：
     ```bash
     yolo detect train data=/root/autodl-tmp/yolov10-main1/yolov10-main/yolo-bun.yaml model=yolov10n.pt epochs=150 batch=16 imgsz=640 device=0
     ```

3. **训练模型（编写脚本形式）**
   - 以下是一个训练模型的脚本示例：
     ```python
     from ultralytics import YOLO
     import torch
     import torch.nn as nn

     def main():
         # 加载模型（使用配置文件）
         model = YOLO("D:\\桌面\\yolov10-main\\ultralytics\\cfg\\models\\v10\\yolov10n.yaml")
         # 训练模型
         train_results = model.train(
             data=r'D:\桌面\yolov10-main\yolo-bvn.yaml',
             epochs=100,
             imgsz=640,
             device=0,
             batch=8,
             optimizer='Adam',
             lr0=0.0005,
             weight_decay=0.0005,
             project='runs/detect',
             name='endtest100',
             close_mosaic=0,
             save=True,
             save_period=-1
         )
         # 评估模型在验证集上的性能
         metrics = model.val(device=0)

     if __name__ == '__main__':
         main()
     ```

### （三）模型转换

1. **保存最佳模型**
   - 在训练过程中，将最好的模型保存下来。

2. **使用 OpenVINO 转换模型**
   - 使用以下代码将模型转换为 OpenVINO 格式：
     ```python
     from ultralytics import YOLO
     import os

     data_path = r'D:\桌面\yolov10-main\yolo-bvn.yaml'  # YAML 文件指向您的数据集
     model = YOLO(r'D:\桌面\yolov10-main\pt\best.pt')  # 加载您的训练模型
     model.export(format="openvino", int8=True, data=data_path)
     ```

### （四）测试与推理

1. **测试脚本流程**
   - 模型加载与初始化：
     - 使用 OpenVINO 的 Core 类加载 YOLOv10 的 IR 格式模型文件（包含 .xml 和 .bin 文件）。
   - 输入与输出设置：
     - 设置测试图像文件夹路径 `input_folder` 和 XML 输出文件夹路径 `output_folder`，确保输出文件夹存在。
   - 图像预处理：
     - 对 `input_folder` 中的图像逐张处理，调整大小、归一化等操作。
   - 推理执行：
     - 处理后的图像通过模型进行推理，返回包含检测框、类别和置信度的预测结果。
   - 非极大值抑制（NMS）：
     - 进行 NMS 处理，过滤掉重叠高的框，保留置信度较高的检测框。
   - 检测结果解析与坐标还原：
     - 根据 NMS 结果还原检测框坐标，生成 XML 文件。
   - 结果保存：
     - 保存检测结果至指定输出文件夹，并输出推理总耗时。

2. **示例代码（部分）**
   ```python
   # 加载OpenVINO模型
   core = Core()
   model_ir = core.read_model(model_xml)
   compiled_model = core.compile_model(model_ir, "CPU")
   input_layer = compiled_model.input(0)

   # 遍历输入文件夹中的图像
   for image_file in os.listdir(input_folder):
       image_path = os.path.join(input_folder, image_file)
       image = cv2.imread(image_path)
       # 图像预处理
       resized_image = letterbox(image, (input_width, input_height))[0]
       resized_image = resized_image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
       resized_image = np.ascontiguousarray(resized_image)
       resized_image = resized_image.astype(np.float32) / 255.0
       # 推理
       result = compiled_model([resized_image])[compiled_model.output(0)]
       # NMS处理等后续操作（省略部分代码）
