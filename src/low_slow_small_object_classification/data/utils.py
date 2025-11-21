import os
import re
from typing import List
from pathlib import Path
from dataclasses import dataclass

from src.low_slow_small_object_classification.utils.logger import default_logger as logger


ABNORMAL_BATCH_ID = [1451, 1452, 1457, 1462, 1467, 1469, 1473, 1478, 1484, 1487, 1488, 1490, 1494, 1496, 1497, 1500]

@dataclass
class BatchFile:
    """批次文件信息"""
    batch_num: int         # 航迹批号
    label: int             # 目标类型标签
    raw_file: str          # 原始回波文件路径
    point_file: str        # 点迹文件路径
    track_file: str        # 航迹文件路径


def get_batch_file_list(data_root: str, test: bool = False) -> List[BatchFile]:
    iq_dir = os.path.join(data_root, "原始回波")
    track_dir = os.path.join(data_root, "航迹")
    point_dir = os.path.join(data_root, "点迹")

    if not all(os.path.isdir(d) for d in [iq_dir, track_dir, point_dir]):
        raise ValueError("错误！数据根目录下需包含原始回波、点迹、航迹三个子文件夹。")

    batch_files = []
    # 遍历原始回波文件
    for raw_file in os.listdir(iq_dir):
        if not raw_file.endswith('.dat'):
            continue

        # 解析文件名
        pattern = r'^(\d+)\.dat$' if test else r'^(\d+)_Label_(\d+)\.dat$'
        match = re.match(pattern, raw_file)
        if not match:
            continue

        batch_num = int(match.group(1))
        label = 0 if test else int(match.group(2))

        if batch_num in ABNORMAL_BATCH_ID or label > 4 or label < 0:
            continue

        # 查找对应的点迹和航迹文件
        if test:
            point_pattern = f'PointTracks_{batch_num}_*.txt'
            track_pattern = f'Tracks_{batch_num}_*.txt'
        else:
            point_pattern = f'PointTracks_{batch_num}_{label}_*.txt'
            track_pattern = f'Tracks_{batch_num}_{label}_*.txt'

        point_files = list(Path(point_dir).glob(point_pattern))
        track_files = list(Path(track_dir).glob(track_pattern))

        if point_files and track_files:
            batch_files.append(BatchFile(
                batch_num=batch_num,
                label=label,
                raw_file=os.path.join(iq_dir, raw_file),
                point_file=str(point_files[0]),
                track_file=str(track_files[0])
            ))
        else:
            missing_point = len(point_files) == 0
            missing_track = len(track_files) == 0
            msg = f"警告：批号 {batch_num}、标签 {label} 的"
            if missing_point and missing_track:
                msg += "点迹和航迹文件均未找到，已跳过。"
            elif missing_point:
                msg += "点迹文件未找到，已跳过。"
            else:
                msg += "航迹文件未找到，已跳过。"
            logger.warning(msg)

    if not batch_files:
        raise ValueError("未找到符合命名规则的批量处理文件（需为：航迹批号_Label_目标类型标签.dat）！")

    return batch_files
