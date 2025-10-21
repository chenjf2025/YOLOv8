import os
import json
from pathlib import Path
from ultralytics import YOLO
import logging
from datetime import datetime

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
                "exist_ok": True  # 允许覆盖现有文件
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
                
                # 使用YOLO内置的预测和保存功能
                results = self.model.predict(
                    source=file_path,
                    conf=self.config.get('confidence_threshold', 0.25),
                    save=self.config.get('save', True),  # 使用内置保存功能
                    save_txt=self.config.get('save_txt', True),  # 保存检测结果文本
                    save_conf=self.config.get('save_conf', True),  # 在文本文件中保存置信度
                    project=self.config.get('output_dir', 'output_results'),  # 输出目录
                    exist_ok=self.config.get('exist_ok', True)  # 允许覆盖现有文件
                )
                
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