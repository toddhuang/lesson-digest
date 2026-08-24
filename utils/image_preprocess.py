"""
图片预处理工具
用于去除视频帧中的彩色部分（老师手写），只保留黑色文字（印刷体题目）。
利用板书特征：题目字体黑色，老师手写彩色。
"""

import cv2
import numpy as np
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger("image_preprocess")


def remove_color_keep_black(
    image_path: str,
    output_path: str,
    black_threshold: int = 120,
    saturation_threshold: int = 40,
) -> str:
    """去除彩色部分，只保留黑色（或接近黑色）文字

    原理：
    - 黑色文字：R、G、B 三通道值都很低，且三通道差值小（饱和度低）
    - 彩色手写：某个通道值明显高于其他通道，饱和度高
    - 白色背景：R、G、B 三通道值都很高

    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        black_threshold: 黑色阈值（0-255），通道值低于此值认为是黑色
        saturation_threshold: 饱和度阈值，通道最大差值低于此值认为是无彩色

    Returns:
        输出图片路径
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"无法读取图片: {image_path}")
        return image_path

    # 分离 BGR 通道
    b = img[:, :, 0].astype(int)
    g = img[:, :, 1].astype(int)
    r = img[:, :, 2].astype(int)

    # 计算最大通道和最小通道的差值（饱和度近似）
    max_channel = np.maximum(np.maximum(r, g), b)
    min_channel = np.minimum(np.minimum(r, g), b)
    saturation = max_channel - min_channel

    # 黑色像素条件：
    # 1. 最大通道值低于黑色阈值（整体暗）
    # 2. 饱和度低于饱和度阈值（无彩色，即灰度/黑色）
    is_black = (max_channel < black_threshold) & (saturation < saturation_threshold)

    # 创建白色背景
    result = np.ones_like(img) * 255

    # 保留黑色像素（用原始颜色，通常是黑色）
    result[is_black] = img[is_black]

    cv2.imwrite(output_path, result)
    logger.debug(f"颜色过滤完成: {image_path} -> {output_path} (保留黑色像素: {np.sum(is_black)})")
    return output_path


def preprocess_frame_for_ocr(
    image_path: str,
    output_dir: Optional[str] = None,
    black_threshold: int = 120,
) -> str:
    """对视频帧进行 OCR 预处理（去除彩色手写，只保留黑色题目）

    Args:
        image_path: 输入图片路径
        output_dir: 输出目录，为 None 则覆盖原文件
        black_threshold: 黑色阈值

    Returns:
        处理后的图片路径
    """
    import os

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
    else:
        output_path = image_path  # 覆盖原文件

    return remove_color_keep_black(
        image_path,
        output_path,
        black_threshold=black_threshold,
    )
