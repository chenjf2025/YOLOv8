# 增加中文标签支持
# 通过PIL库在图像上绘制中文标签
import os
import json
from pathlib import Path
from ultralytics import YOLO
import logging
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class YOLOv8Predictor:
    def __init__(self, config_path="config.json"):
        """
        初始化YOLOv8预测器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self.load_config()
        self.model = None
        self.processed_files = set()
        
        # 加载中文标签映射
        self.chinese_labels = self.load_chinese_labels()
        
        # 设置日志
        self.setup_logging()
        
        # 加载模型
        self.load_model()
        
        # 加载已处理文件记录
        self.load_processed_files()
    
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 验证必要配置项
            required_keys = ['input_dir', 'output_dir', 'record_file']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"配置文件中缺少必要的键: {key}")
            
            # 创建输入目录（如果不存在）
            os.makedirs(config['input_dir'], exist_ok=True)
            
            return config
        except FileNotFoundError:
            # 如果配置文件不存在，创建默认配置
            default_config = {
                "input_dir": "input_files",
                "output_dir": "output_results",
                "record_file": "processed_files.json",
                "model_path": "yolov8n.pt",
                "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"],
                "confidence_threshold": 0.25,
                "save": True,
                "save_txt": True,
                "save_conf": True,
                "exist_ok": True,
                "use_chinese_labels": True,  # 是否使用中文标签
                "font_path": "simhei.ttf",  # 中文字体文件路径
                "font_size": 20,  # 字体大小
                "label_position": "center"  # 标签位置：center/top/bottom
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            
            print(f"已创建默认配置文件: {self.config_path}")
            return default_config
    
    def load_chinese_labels(self):
        """加载中文标签映射"""
        # COCO数据集类别中英文对照
        chinese_labels = {
            0: "人", 1: "自行车", 2: "汽车", 3: "摩托车", 4: "飞机", 
            5: "公交车", 6: "火车", 7: "卡车", 8: "船", 9: "交通灯",
            10: "消防栓", 11: "停车标志", 12: "停车计时器", 13: "长椅", 14: "鸟",
            15: "猫", 16: "狗", 17: "马", 18: "羊", 19: "牛",
            20: "大象", 21: "熊", 22: "斑马", 23: "长颈鹿", 24: "背包",
            25: "雨伞", 26: "手提包", 27: "领带", 28: "行李箱", 29: "飞盘",
            30: "滑雪板", 31: "滑雪板", 32: "运动球", 33: "风筝", 34: "棒球棒",
            35: "棒球手套", 36: "滑板", 37: "冲浪板", 38: "网球拍", 39: "瓶子",
            40: "葡萄酒杯", 41: "杯子", 42: "叉子", 43: "刀", 44: "勺子",
            45: "碗", 46: "香蕉", 47: "苹果", 48: "三明治", 49: "橙子",
            50: "西兰花", 51: "胡萝卜", 52: "热狗", 53: "披萨", 54: "甜甜圈",
            55: "蛋糕", 56: "椅子", 57: "沙发", 58: "盆栽", 59: "床",
            60: "餐桌", 61: "厕所", 62: "电视", 63: "笔记本电脑", 64: "鼠标",
            65: "遥控器", 66: "键盘", 67: "手机", 68: "微波炉", 69: "烤箱",
            70: "烤面包机", 71: "水槽", 72: "冰箱", 73: "书", 74: "时钟",
            75: "花瓶", 76: "剪刀", 77: "泰迪熊", 78: "吹风机", 79: "牙刷"
        }
        return chinese_labels
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """加载YOLOv8模型"""
        try:
            model_path = self.config.get('model_path', 'yolov8n.pt')
            self.logger.info(f"正在加载模型: {model_path}")
            
            self.model = YOLO(model_path)
            self.logger.info("模型加载成功")
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def load_processed_files(self):
        """加载已处理文件记录"""
        record_file = self.config['record_file']
        try:
            if os.path.exists(record_file):
                with open(record_file, 'r', encoding='utf-8') as f:
                    self.processed_files = set(json.load(f))
                self.logger.info(f"已加载 {len(self.processed_files)} 个已处理文件记录")
            else:
                self.logger.info("未找到已处理文件记录，将创建新记录")
        except Exception as e:
            self.logger.warning(f"加载已处理文件记录失败: {str(e)}，将创建新记录")
            self.processed_files = set()
    
    def save_processed_files(self):
        """保存已处理文件记录"""
        try:
            record_file = self.config['record_file']
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.processed_files), f, indent=4, ensure_ascii=False)
            self.logger.info(f"已保存 {len(self.processed_files)} 个已处理文件记录")
        except Exception as e:
            self.logger.error(f"保存已处理文件记录失败: {str(e)}")
    
    def get_supported_files(self):
        """获取支持的文件列表"""
        input_dir = self.config['input_dir']
        supported_formats = self.config.get('supported_formats', [".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"])
        
        if not os.path.exists(input_dir):
            self.logger.warning(f"输入目录不存在: {input_dir}")
            return []
        
        all_files = []
        for format in supported_formats:
            all_files.extend(Path(input_dir).rglob(f"*{format}"))
            # 同时查找大写格式
            all_files.extend(Path(input_dir).rglob(f"*{format.upper()}"))
        
        # 去重并转换为字符串路径
        file_paths = list(set(str(f) for f in all_files))
        
        # 过滤掉已处理的文件
        unprocessed_files = [f for f in file_paths if f not in self.processed_files]
        
        self.logger.info(f"找到 {len(file_paths)} 个支持的文件，其中 {len(unprocessed_files)} 个未处理")
        
        return unprocessed_files
    
    def draw_boxes_with_chinese(self, image, boxes, class_ids, confidences):
        """在图像上绘制边界框和中文标签"""
        # 将OpenCV图像转换为PIL图像
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        
        # 获取图像尺寸
        img_width, img_height = image_pil.size
        
        # 尝试加载中文字体
        font_path = self.config.get('font_path', 'simhei.ttf')
        font_size = self.config.get('font_size', 20)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            self.logger.warning(f"无法加载字体 {font_path}，使用默认字体")
            font = ImageFont.load_default()
        
        # 为每个检测到的物体绘制边界框和标签
        for i, (box, class_id, confidence) in enumerate(zip(boxes, class_ids, confidences)):
            # 获取边界框坐标
            x1, y1, x2, y2 = box
            
            # 获取类别名称
            class_name_en = self.model.names[int(class_id)]
            class_name_cn = self.chinese_labels.get(int(class_id), class_name_en)
            
            # 创建标签文本
            label = f"{class_name_cn} {confidence:.2f}"
            
            # 计算文本大小
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 计算标签位置 - 放在边界框内部的中间位置
            label_position = self.config.get('label_position', 'center')
            
            if label_position == 'center':
                # 计算边界框中心
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2
                
                # 计算标签位置 - 居中
                label_x = box_center_x - text_width / 2
                label_y = box_center_y - text_height / 2
                
                # 确保标签不会超出边界框
                label_x = max(x1, min(label_x, x2 - text_width))
                label_y = max(y1, min(label_y, y2 - text_height))
                
            elif label_position == 'top':
                # 放在边界框顶部
                label_x = x1
                label_y = y1
                
                # 如果顶部空间不够，放在底部
                if y1 + text_height + 5 > y2:
                    label_y = y2 - text_height - 5
            else:  # bottom
                # 放在边界框底部
                label_x = x1
                label_y = y2 - text_height - 5
                
                # 如果底部空间不够，放在顶部
                if label_y < y1:
                    label_y = y1
            
            # 确保标签不会超出图像边界
            label_x = max(0, min(label_x, img_width - text_width - 10))
            label_y = max(0, min(label_y, img_height - text_height - 5))
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            
            # 绘制标签背景 - 半透明效果
            label_bg = [label_x - 5, label_y - 2, label_x + text_width + 5, label_y + text_height + 2]
            
            # 创建一个临时图像来绘制半透明背景
            temp_img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            temp_draw.rectangle(label_bg, fill=(0, 0, 0, 180))  # 黑色半透明背景
            
            # 将临时图像合并到原图像
            image_pil = Image.alpha_composite(image_pil.convert('RGBA'), temp_img).convert('RGB')
            draw = ImageDraw.Draw(image_pil)
            
            # 绘制标签文本
            draw.text((label_x, label_y), label, fill="white", font=font)
        
        # 将PIL图像转换回OpenCV格式
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    def process_image_with_chinese_labels(self, image_path, output_dir):
        """处理图片并添加中文标签"""
        try:
            # 读取原始图像
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"无法读取图像: {image_path}")
                return False
            
            # 运行预测
            results = self.model.predict(
                source=image_path,
                conf=self.config.get('confidence_threshold', 0.25),
                save=False
            )
            
            # 提取检测结果
            boxes = []
            class_ids = []
            confidences = []
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        boxes.append([x1, y1, x2, y2])
                        
                        # 获取类别ID
                        class_id = box.cls[0].cpu().numpy()
                        class_ids.append(class_id)
                        
                        # 获取置信度
                        confidence = box.conf[0].cpu().numpy()
                        confidences.append(confidence)
            
            # 在图像上绘制中文标签
            if boxes:
                image_with_boxes = self.draw_boxes_with_chinese(image, boxes, class_ids, confidences)
            else:
                image_with_boxes = image
            
            # 保存结果图像
            filename = Path(image_path).stem
            relative_path = Path(image_path).relative_to(self.config['input_dir'])
            output_subdir = Path(output_dir) / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_subdir / f"{filename}_detected.jpg"
            cv2.imwrite(str(output_path), image_with_boxes)
            self.logger.info(f"保存检测结果图片: {output_path}")
            
            # 保存检测数据 - 修复JSON序列化问题
            if self.config.get('save_txt', True):
                detection_data = []
                for i, (box, class_id, confidence) in enumerate(zip(boxes, class_ids, confidences)):
                    # 将所有numpy类型转换为Python原生类型
                    detection_data.append({
                        "box_index": i,
                        "class_id": int(class_id),
                        "class_en": self.model.names[int(class_id)],
                        "class_cn": self.chinese_labels.get(int(class_id), self.model.names[int(class_id)]),
                        "confidence": float(confidence),  # 转换为Python float
                        "coordinates": [float(coord) for coord in box]  # 转换为Python float列表
                    })
                
                if detection_data:
                    data_file = output_subdir / f"{filename}_detections.json"
                    with open(data_file, 'w', encoding='utf-8') as f:
                        json.dump(detection_data, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"保存检测数据: {data_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理图片失败 {image_path}: {str(e)}")
            return False
    
    def is_image_file(self, file_path):
        """判断是否为图片文件"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        return Path(file_path).suffix.lower() in image_extensions
    
    def is_video_file(self, file_path):
        """判断是否为视频文件"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        return Path(file_path).suffix.lower() in video_extensions
    
    def process_files(self):
        """处理所有未处理的文件"""
        files_to_process = self.get_supported_files()
        
        if not files_to_process:
            self.logger.info("没有需要处理的文件")
            return
        
        self.logger.info(f"开始处理 {len(files_to_process)} 个文件")
        
        success_count = 0
        for file_path in files_to_process:
            try:
                self.logger.info(f"处理文件: {file_path}")
                
                # 根据文件类型和配置选择处理方式
                if self.is_image_file(file_path) and self.config.get('use_chinese_labels', True):
                    # 使用自定义处理方式添加中文标签
                    output_dir = self.config['output_dir']
                    success = self.process_image_with_chinese_labels(file_path, output_dir)
                else:
                    # 使用YOLO内置的预测和保存功能
                    results = self.model.predict(
                        source=file_path,
                        conf=self.config.get('confidence_threshold', 0.25),
                        save=self.config.get('save', True),
                        save_txt=self.config.get('save_txt', True),
                        save_conf=self.config.get('save_conf', True),
                        project=self.config.get('output_dir', 'output_results'),
                        exist_ok=self.config.get('exist_ok', True)
                    )
                    success = True
                
                if success:
                    # 记录已处理文件
                    self.processed_files.add(file_path)
                    success_count += 1
                    
                    # 每处理完一个文件就保存记录，防止程序中断时丢失进度
                    self.save_processed_files()
                    
                    self.logger.info(f"成功处理: {file_path}")
                
            except Exception as e:
                self.logger.error(f"处理文件失败 {file_path}: {str(e)}")
                continue
        
        self.logger.info(f"处理完成: 成功 {success_count}/{len(files_to_process)} 个文件")
    
    def run(self):
        """运行预测器"""
        self.logger.info("开始YOLOv8批量预测")
        start_time = datetime.now()
        
        try:
            self.process_files()
        except KeyboardInterrupt:
            self.logger.info("用户中断处理")
        except Exception as e:
            self.logger.error(f"处理过程中发生错误: {str(e)}")
        finally:
            # 最终保存处理记录
            self.save_processed_files()
            
            end_time = datetime.now()
            duration = end_time - start_time
            self.logger.info(f"处理结束，总耗时: {duration}")

def main():
    """主函数"""
    # 可以在这里修改配置文件路径
    config_file = "yolov8_predictor_config.json"
    
    predictor = YOLOv8Predictor(config_file)
    predictor.run()

if __name__ == "__main__":
    main()