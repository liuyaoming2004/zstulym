from ultralytics import YOLO
import os
import time

if __name__ == '__main__':
    # 开始计时
    start_time = time.time()

    # 加载训练好的模型
    model = YOLO(r'D:\桌面\yolov10-main\2022337621219best.pt')

    # 设置要检测的图像文件夹路径
    image_folder = "test"  # 替换为你要检测的图像文件夹路径

    # 遍历文件夹中的所有图像文件
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)

        # 确保只处理图像文件
        if image_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            print(f"Processing image: {image_path}")

            # 在图像上执行对象检测
            results = model(image_path, save=True)

    # 结束计时
    end_time = time.time()

    # 计算并打印总时间
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")