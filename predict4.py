#识别特定的时钟时间

import os
import cv2
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
import re
from ultralytics import YOLO
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import math

class ClockTimeReader:
    def __init__(self, config_path="clock_config.json"):
        """
        初始化时钟时间读取器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self.load_config()
        
        # 设置日志
        self.setup_logging()
        
        # 加载YOLO模型用于检测时钟
        self.load_model()
        
        # 结果存储
        self.results = []
    
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 验证必要配置项
            required_keys = ['input_dir', 'output_dir', 'result_file']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"配置文件中缺少必要的键: {key}")
            
            # 创建输出目录（如果不存在）
            os.makedirs(config['output_dir'], exist_ok=True)
            
            return config
        except FileNotFoundError:
            # 如果配置文件不存在，创建默认配置
            default_config = {
                "input_dir": "clock_images",
                "output_dir": "clock_results",
                "result_file": "clock_times.json",
                "model_path": "yolov8n.pt",
                "supported_formats": [".jpg", ".jpeg", ".png", ".bmp"],
                "confidence_threshold": 0.25,
                "save_annotated_images": True,
                "tesseract_path": "C:/Program Files/Tesseract-OCR/tesseract.exe",  # Windows默认路径
                "clock_class_id": 74  # COCO数据集中时钟的类别ID
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            
            print(f"已创建默认配置文件: {self.config_path}")
            return default_config
    
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
        """加载YOLO模型"""
        try:
            model_path = self.config.get('model_path', 'yolov8n.pt')
            self.logger.info(f"正在加载模型: {model_path}")
            
            self.model = YOLO(model_path)
            self.logger.info("模型加载成功")
            
            # 设置Tesseract路径（如果提供）
            tesseract_path = self.config.get('tesseract_path')
            if tesseract_path and os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                self.logger.info(f"已设置Tesseract路径: {tesseract_path}")
                
            # 设置 TESSDATA_PREFIX 环境变量，指向包含 'tessdata' 文件夹的目录
            tessdata_dir = r"C:\\Program Files\\Tesseract-OCR\\tessdata"  # 请根据您电脑上的实际路径修改
            os.environ['TESSDATA_PREFIX'] = tessdata_dir
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def get_supported_files(self):
        """获取支持的文件列表"""
        input_dir = self.config['input_dir']
        supported_formats = self.config.get('supported_formats', [".jpg", ".jpeg", ".png", ".bmp"])
        
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
        
        self.logger.info(f"找到 {len(file_paths)} 个支持的文件")
        
        return file_paths
    
    def detect_clocks(self, image_path):
        """检测图像中的时钟"""
        try:
            # 运行预测
            results = self.model.predict(
                source=image_path,
                conf=self.config.get('confidence_threshold', 0.25),
                save=False
            )
            
            # 提取时钟检测结果
            clock_boxes = []
            clock_confidences = []
            
            clock_class_id = self.config.get('clock_class_id', 74)
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        if class_id == clock_class_id:
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            clock_boxes.append([int(x1), int(y1), int(x2), int(y2)])
                            
                            # 获取置信度
                            confidence = float(box.conf[0].cpu().numpy())
                            clock_confidences.append(confidence)
            
            return clock_boxes, clock_confidences
            
        except Exception as e:
            self.logger.error(f"检测时钟失败 {image_path}: {str(e)}")
            return [], []
    
    def preprocess_clock_region(self, clock_roi):
        """预处理时钟区域以改善OCR识别"""
        try:
            # 转换为灰度图
            if len(clock_roi.shape) == 3:
                gray = cv2.cvtColor(clock_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = clock_roi
            
            # 应用高斯模糊减少噪声
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 二值化处理
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 形态学操作去除噪声
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 放大图像以提高OCR精度
            scale_factor = 2
            height, width = cleaned.shape
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resized = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            return resized
            
        except Exception as e:
            self.logger.warning(f"预处理时钟区域失败: {str(e)}")
            return clock_roi
    
    def extract_digital_time(self, clock_roi):
        """从电子时钟区域提取时间"""
        try:
            # 预处理时钟区域
            processed_roi = self.preprocess_clock_region(clock_roi)
            
            # 使用Tesseract进行OCR识别
            # 配置Tesseract参数
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789:'
            text = pytesseract.image_to_string(processed_roi, config=custom_config)
            
            # 清理和提取时间
            time_patterns = [
                r'(\d{1,2}):(\d{2}):(\d{2})',  # HH:MM:SS
                r'(\d{1,2}):(\d{2})',          # HH:MM
                r'(\d{1,2})\.(\d{2})',         # HH.MM
                r'(\d{1,2})\s+(\d{2})',        # HH MM
                r'(\d{4})',                    # HHMM (24小时制)
                r'(\d{2})(\d{2})',             # HHMM (分开)
            ]
            
            for pattern in time_patterns:
                match = re.search(pattern, text)
                if match:
                    if len(match.groups()) == 3:  # HH:MM:SS
                        hour, minute, second = match.groups()
                        return f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}"
                    elif len(match.groups()) == 2:  # HH:MM
                        hour, minute = match.groups()
                        return f"{int(hour):02d}:{int(minute):02d}"
                    elif len(match.groups()) == 1:  # HHMM
                        time_str = match.group(1)
                        if len(time_str) == 4:
                            return f"{time_str[:2]}:{time_str[2:]}"
            
            # 如果正则匹配失败，尝试直接提取数字
            digits = re.findall(r'\d+', text)
            if len(digits) >= 2:
                hour = int(digits[0]) % 24
                minute = int(digits[1]) % 60
                return f"{hour:02d}:{minute:02d}"
            
            return "未知"
            
        except Exception as e:
            self.logger.warning(f"提取数字时间失败: {str(e)}")
            return "未知"
    
    def detect_clock_center_and_hands(self, clock_roi):
        """检测指针式时钟的中心和指针"""
        try:
            # 转换为灰度图
            if len(clock_roi.shape) == 3:
                gray = cv2.cvtColor(clock_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = clock_roi
            
            # 边缘检测
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # 霍夫圆检测寻找时钟表盘
            circles = cv2.HoughCircles(
                edges, 
                cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=20,
                param1=50,
                param2=30,
                minRadius=min(clock_roi.shape[:2])//4,
                maxRadius=min(clock_roi.shape[:2])//2
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                if len(circles) > 0:
                    # 取最大的圆
                    circle = circles[0]
                    center_x, center_y, radius = circle
                    
                    # 检测直线（指针）
                    lines = cv2.HoughLinesP(
                        edges, 
                        1, 
                        np.pi/180, 
                        threshold=30,
                        minLineLength=radius//2,
                        maxLineGap=10
                    )
                    
                    return (center_x, center_y, radius), lines
            
            return None, None
            
        except Exception as e:
            self.logger.warning(f"检测时钟中心和指针失败: {str(e)}")
            return None, None
    
    def calculate_analog_time(self, center, lines):
        """根据指针位置计算模拟时钟时间"""
        try:
            center_x, center_y, radius = center
            
            if lines is None or len(lines) == 0:
                return "未知"
            
            # 计算每条线的角度
            angles = []
            lengths = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # 计算线段中点
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                # 计算线段相对于中心的角度
                dx = mid_x - center_x
                dy = mid_y - center_y
                angle = math.degrees(math.atan2(dy, dx))
                
                # 转换为时钟角度（0度在12点位置，顺时针增加）
                clock_angle = (angle + 90) % 360
                if clock_angle < 0:
                    clock_angle += 360
                
                # 计算线段长度
                length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                angles.append(clock_angle)
                lengths.append(length)
            
            # 找到时针和分针（假设最长的两根指针）
            if len(angles) < 2:
                return "未知"
            
            # 按长度排序
            sorted_indices = np.argsort(lengths)[::-1]  # 从长到短
            
            # 取最长的两根作为时针和分针
            hour_hand_idx = sorted_indices[0]
            minute_hand_idx = sorted_indices[1]
            
            hour_angle = angles[hour_hand_idx]
            minute_angle = angles[minute_hand_idx]
            
            # 计算时间
            # 时针：每小时30度，每分钟0.5度
            hour = int(hour_angle / 30) % 12
            hour_minute_component = (hour_angle % 30) / 30 * 60
            hour = int(hour + hour_minute_component / 60)
            
            # 分针：每分钟6度
            minute = int(minute_angle / 6)
            
            # 处理边界情况
            if hour == 0:
                hour = 12
            if hour > 12:
                hour -= 12
            
            return f"{int(hour):02d}:{int(minute):02d}"
            
        except Exception as e:
            self.logger.warning(f"计算模拟时间失败: {str(e)}")
            return "未知"
    
    def extract_analog_time(self, clock_roi):
        """从指针式时钟区域提取时间"""
        try:
            # 检测时钟中心和指针
            center, lines = self.detect_clock_center_and_hands(clock_roi)
            
            if center is None:
                return "未知"
            
            # 计算时间
            return self.calculate_analog_time(center, lines)
            
        except Exception as e:
            self.logger.warning(f"提取模拟时间失败: {str(e)}")
            return "未知"
    
    def determine_clock_type_and_extract_time(self, clock_roi):
        """确定时钟类型并提取时间"""
        # 首先尝试提取数字时间
        digital_time = self.extract_digital_time(clock_roi)
        
        if digital_time != "未知":
            return "电子时钟", digital_time
        
        # 如果数字时间提取失败，尝试提取模拟时间
        analog_time = self.extract_analog_time(clock_roi)
        
        if analog_time != "未知":
            return "指针式时钟", analog_time
        
        return "未知类型", "未知"
    
    def draw_annotations(self, image, clock_boxes, clock_times, clock_types):
        """在图像上绘制检测结果和识别的时间"""
        try:
            # 将OpenCV图像转换为PIL图像
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image_pil)
            
            # 尝试加载中文字体
            font_path = "simhei.ttf"  # 默认字体路径
            font_size = 20
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                self.logger.warning(f"无法加载字体 {font_path}，使用默认字体")
                font = ImageFont.load_default()
            
            # 为每个检测到的时钟绘制边界框和标签
            for i, (box, time, clock_type) in enumerate(zip(clock_boxes, clock_times, clock_types)):
                x1, y1, x2, y2 = box
                
                # 创建标签文本
                label = f"{clock_type}: {time}"
                
                # 计算文本大小
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 绘制边界框
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                # 绘制标签背景
                draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 10, y1], fill="red")
                
                # 绘制标签文本
                draw.text((x1 + 5, y1 - text_height - 2), label, fill="white", font=font)
            
            # 将PIL图像转换回OpenCV格式
            return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            self.logger.warning(f"绘制标注失败: {str(e)}")
            return image
    
    def process_image(self, image_path):
        """处理单张图片"""
        try:
            self.logger.info(f"处理图片: {image_path}")
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"无法读取图像: {image_path}")
                return False
            
            # 检测时钟
            clock_boxes, clock_confidences = self.detect_clocks(image_path)
            
            if not clock_boxes:
                self.logger.info(f"在 {image_path} 中未检测到时钟")
                # 仍然记录结果，但时间为未知
                self.results.append({
                    "filename": os.path.basename(image_path),
                    "clock_count": 0,
                    "clocks": []
                })
                return True
            
            self.logger.info(f"在 {image_path} 中检测到 {len(clock_boxes)} 个时钟")
            
            # 处理每个检测到的时钟
            clock_times = []
            clock_types = []
            
            for i, box in enumerate(clock_boxes):
                x1, y1, x2, y2 = box
                
                # 提取时钟区域
                clock_roi = image[y1:y2, x1:x2]
                
                # 确定时钟类型并提取时间
                clock_type, time = self.determine_clock_type_and_extract_time(clock_roi)
                
                clock_times.append(time)
                clock_types.append(clock_type)
                
                self.logger.info(f"时钟 {i+1}: {clock_type}, 时间: {time}")
            
            # 保存标注图像
            if self.config.get('save_annotated_images', True):
                annotated_image = self.draw_annotations(image, clock_boxes, clock_times, clock_types)
                
                output_dir = self.config['output_dir']
                filename = Path(image_path).stem
                relative_path = Path(image_path).relative_to(self.config['input_dir'])
                output_subdir = Path(output_dir) / relative_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                output_path = output_subdir / f"{filename}_annotated.jpg"
                cv2.imwrite(str(output_path), annotated_image)
                self.logger.info(f"保存标注图像: {output_path}")
            
            # 记录结果
            clocks_info = []
            for i, (box, confidence, clock_type, time) in enumerate(zip(clock_boxes, clock_confidences, clock_types, clock_times)):
                clocks_info.append({
                    "clock_id": i+1,
                    "bounding_box": box,
                    "confidence": confidence,
                    "type": clock_type,
                    "time": time
                })
            
            self.results.append({
                "filename": os.path.basename(image_path),
                "clock_count": len(clock_boxes),
                "clocks": clocks_info
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理图片失败 {image_path}: {str(e)}")
            return False
    
    def save_results(self):
        """保存结果到文件"""
        try:
            result_file = self.config['result_file']
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=4, ensure_ascii=False)
            self.logger.info(f"结果已保存到: {result_file}")
        except Exception as e:
            self.logger.error(f"保存结果失败: {str(e)}")
    
    def process_all_images(self):
        """处理所有图片"""
        files_to_process = self.get_supported_files()
        
        if not files_to_process:
            self.logger.info("没有需要处理的文件")
            return
        
        self.logger.info(f"开始处理 {len(files_to_process)} 个文件")
        
        success_count = 0
        for file_path in files_to_process:
            try:
                success = self.process_image(file_path)
                if success:
                    success_count += 1
            except Exception as e:
                self.logger.error(f"处理文件失败 {file_path}: {str(e)}")
                continue
        
        self.logger.info(f"处理完成: 成功 {success_count}/{len(files_to_process)} 个文件")
        
        # 保存结果
        self.save_results()
    
    def run(self):
        """运行时钟时间读取器"""
        self.logger.info("开始时钟时间读取")
        start_time = datetime.now()
        
        try:
            self.process_all_images()
        except KeyboardInterrupt:
            self.logger.info("用户中断处理")
        except Exception as e:
            self.logger.error(f"处理过程中发生错误: {str(e)}")
        finally:
            # 最终保存结果
            self.save_results()
            
            end_time = datetime.now()
            duration = end_time - start_time
            self.logger.info(f"处理结束，总耗时: {duration}")

def main():
    """主函数"""
    # 可以在这里修改配置文件路径
    config_file = "clock_config.json"
    
    clock_reader = ClockTimeReader(config_file)
    clock_reader.run()

if __name__ == "__main__":
    main()