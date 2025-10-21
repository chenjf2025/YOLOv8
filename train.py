from ultralytics import YOLO

# 1. 加载一个预训练的 YOLOv8 模型
# 我们选择 yolo-v8n (nano) 作为起点，因为它速度快，适合边缘部署。
# 如果需要更高的精度，可以尝试 yolo-v8s (small)。
model = YOLO('yolov8n.pt') 

# 2. 模型训练配置
# data: 您的配置文件路径
# epochs: 训练轮数（初次可以先设置小一些，如 50）
# imgsz: 图像尺寸（640x640 是标准尺寸，您也可以根据数据调整）
# batch: 批次大小（取决于您的显存，如果内存不足可以减小，如 8 或 4）
results = model.train(
    data='oil_cup_data.yaml', 
    epochs=100, 
    imgsz=640, 
    batch=16,
    name='oil_cup_v1' # 训练结果将保存在 runs/detect/oil_cup_v1 文件夹
)

print("训练完成，结果保存在 runs/detect/oil_cup_v1")