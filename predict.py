#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 图片/视频批量推理 + 断点续跑工具
usage:
    python predict.py  --cfg cfg.json
or
    python predict.py  --src /data/images --dst /result --weights yolov8n.pt
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import cv2
from ultralytics import YOLO

# 支持的后缀
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
VID_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("YOLOv8-Batch")


# ========= 通用工具 =========
def load_done(done_log: Path):
    """读取已完成列表"""
    if not done_log.exists():
        return set()
    with done_log.open("r", encoding="utf8") as f:
        return {line.strip() for line in f if line.strip()}


def append_done(done_log: Path, abs_path: str):
    """追加已完成文件"""
    with done_log.open("a", encoding="utf8") as f:
        f.write(abs_path + "\n")


def list_media_files(src_dir: Path):
    """递归收集图片/视频"""
    files = []
    for p in src_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (IMG_EXT | VID_EXT):
            files.append(p)
    return sorted(files)


# ========= 推理 =========
def infer_to_txt(model: YOLO, path: Path, save_dir: Path, conf_thres: float = 0.25):
    """
    对单文件进行推理，结果保存为与输入文件同名 txt
    格式：cls x_center y_center width height  (归一化)
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    txt_path = save_dir / (path.stem + ".txt")

    is_video = path.suffix.lower() in VID_EXT
    if is_video:
        cap = cv2.VideoCapture(str(path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        all_boxes = []
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=conf_thres, verbose=False)
            for r in results:
                if r.boxes is not None:
                    for b in r.boxes:
                        cls = int(b.cls)
                        xywhn = b.xywhn[0].tolist()  # 归一化
                        all_boxes.append(f"{cls} " + " ".join(f"{v:.6f}" for v in xywhn))
            frame_id += 1
        cap.release()
        # 视频：把每一帧检测全部写入（如需要可改成只写最大置信等策略）
        txt_path.write_text("\n".join(all_boxes), encoding="utf8")
    else:
        results = model(str(path), conf=conf_thres, verbose=False)
        lines = []
        for r in results:
            if r.boxes is not None:
                for b in r.boxes:
                    cls = int(b.cls)
                    xywhn = b.xywhn[0].tolist()
                    lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in xywhn))
        txt_path.write_text("\n".join(lines), encoding="utf8")

    return txt_path


# ========= 主入口 =========
def main():
    parser = argparse.ArgumentParser(description="YOLOv8 批量推理 + 断点续跑")
    parser.add_argument("--cfg", type=str, help="json 配置文件（优先级低于命令行）")
    parser.add_argument("--src", type=str, help="待预测目录")
    parser.add_argument("--dst", type=str, help="结果保存目录")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信阈值")
    parser.add_argument("--done", type=str, default="done.log", help="已完成列表文件")
    args = parser.parse_args()

    # 1. 参数优先级：命令行 > cfg.json
    if args.cfg:
        with open(args.cfg, "r", encoding="utf8") as f:
            cfg = json.load(f)
    else:
        cfg = {}
    src_dir = Path(args.src or cfg.get("src"))
    dst_dir = Path(args.dst or cfg.get("dst"))
    weights = Path(args.weights or cfg.get("weights", "yolov8n.pt"))
    done_log = Path(args.done or cfg.get("done", "done.log"))
    conf_thres = args.conf or cfg.get("conf", 0.25)

    if not src_dir or not dst_dir:
        log.error("必须指定 --src / --dst 或在 cfg.json 中配置")
        sys.exit(1)

    # 2. 加载模型（自动下载）
    if not weights.exists():
        log.info(f"本地未找到 {weights}，开始自动下载...")
    model = YOLO(str(weights))

    # 3. 扫描文件 & 断点
    all_files = list_media_files(src_dir)
    done_set = load_done(done_log)
    todo = [f for f in all_files if str(f.absolute()) not in done_set]
    log.info(f"共发现 {len(all_files)} 个媒体文件，已跳过 {len(all_files)-len(todo)} 个已完成文件，剩余 {len(todo)} 个待推理")

    # 4. 批量推理
    for idx, path in enumerate(todo, 1):
        abs_path = str(path.absolute())
        log.info(f"[{idx:>4}/{len(todo)}] 推理中：{path.name}")
        try:
            infer_to_txt(model, path, dst_dir, conf_thres)
            append_done(done_log, abs_path)
        except Exception as e:
            log.exception(f"推理失败：{path} | 错误：{e}")

    log.info("全部完成！")


if __name__ == "__main__":
    main()