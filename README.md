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
   ```
### （五）结果可视化
1. **可视化流程**
   - 定义类别名称与颜色的映射关系。
   - 遍历 XML 文件夹，读取每个 XML 文件并解析检测框位置、置信度、类别等信息。
   - 绘制检测框并添加标签，保存至指定的输出文件夹中。
  
2. **示例代码（部分）**
   ```python
   class_names = {0: "标题", 1: "正文", 2: "图片", 3: "图表"}
   colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]

   input_folder = "test_images"
   xml_folder = "xml_results"
   output_folder = "visualized_results"
   os.makedirs(output_folder, exist_ok=True)

   for xml_file in os.listdir(xml_folder):
      xml_path = os.path.join(xml_folder, xml_file)
      image_file = xml_file.replace(".xml", ".jpg")
      image_path = os.path.join(input_folder, image_file)
      image = cv2.imread(image_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.iter('object'):
        class_id = int(obj.find('name').text)
        confidence = float(obj.find('confidence').text)
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        label = f"{class_names[class_id]} {confidence:.2f}"
        color = colors[class_id]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, image)
    print(f"Processed {image_file} and saved to {output_path}")
   ```
## 三、注意事事项
 - 数据集的标注质量 对模型训练效果有很大影响，确保标注准确、完整。
 - 在模型训练过程中，可以根据实际情况调整训练参数，如 epochs、batch、lr0 等，以优化模型性能。
 - 模型转换为 OpenVINO 格式后，可以根据实际硬件环境选择合适的设备（如 CPU、GPU 等）进行推理，以提高推理速度。
 - 在结果可视化过程中，如果遇到图像显示异常等问题，可以检查图像路径、类别映射等设置是否正确。

     
   
