import cv2
import xml.etree.ElementTree as ET
import os

# 类别名称映射和对应的颜色
class_names = {0: "Header", 1: "Title", 2: "Text", 3: "Figure", 4: "Foot"}
colors = {0: (0, 255, 255), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}

# 输入图片和XML文件夹路径
input_folder = 'test'
xml_folder = 'zyoutput'
output_folder = 'zyvisualized_output'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历XML文件
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(xml_folder, xml_file)
        image_file = os.path.splitext(xml_file)[0] + ".jpg"
        image_path = os.path.join(input_folder, image_file)

        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片：{image_path}")
            continue

        # 解析XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 遍历每个检测框
        for detection in root.findall("detection"):
            x_min = int(detection.get("x_min"))
            y_min = int(detection.get("y_min"))
            x_max = int(detection.get("x_max"))
            y_max = int(detection.get("y_max"))
            confidence = float(detection.get("confidence"))
            class_id = int(detection.get("class_id"))

            # 获取类别名称和颜色
            label = class_names.get(class_id, "Unknown")
            color = colors.get(class_id, (0, 0, 0))

            # 绘制边界框
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

            # 设置标签文字及背景
            text = f"{label}: {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_x, text_y = x_min, y_min - 10

            # 增加标签背景
            cv2.rectangle(image, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y), color, -1)
            cv2.putText(image, text, (text_x, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 保存可视化结果
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, image)
        print(f"可视化完成：{output_path}")
