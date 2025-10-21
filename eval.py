from ultralytics import YOLO

# 1. 加载训练好的模型 (请替换为实际路径)
# 训练好的模型路径通常是：runs/detect/oil_cup_v1/weights/best.pt
model_path = 'runs/detect/oil_cup_v1/weights/best.pt'
model = YOLO(model_path) 

# 2. 在测试集上评估模型
# 'data' 指向您的配置文件
# 'split' 设置为 'test'
metrics = model.val(data='oil_cup_data.yaml', split='test') 

# 3. 打印关键指标
# mAP50 是目标检测中 F1 Score 的常用替代指标
print(f"mAP50 (Mean Average Precision at IoU=0.5): {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

# 目标：metrics.box.map50 应该大于 0.8
if metrics.box.map50 > 0.8:
    print("恭喜！油位识别模型性能达标！")
else:
    print("模型性能仍在提升中，请考虑增加数据、调整参数或继续训练。")