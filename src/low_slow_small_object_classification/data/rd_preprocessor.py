import os
import re
import struct
import numpy as np
import polars as pl
from scipy import signal
from scipy.fft import fft, fftshift
from dataclasses import dataclass
from io import BufferedReader
from pathlib import Path
from typing import List, Dict, Any, Optional

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


@dataclass
class Parameters:
    """雷达参数"""
    e_scan_az: np.float64       # 方位角
    track_no_info: np.ndarray   # 航迹信息
    freq: np.float64            # 频率
    cpi_count: np.uint32        # CPI流水号
    prt_num: np.uint32          # PRT数目
    prt: np.float64             # PRT值
    data_length: np.uint32      # 距离维采样点数


def read_raw_data(fid: BufferedReader):
    """
    读取原始回波数据
    :param fid: 文件句柄
    :return: 参数对象和数据数组
    """
    frame_head = 0xFA55FA55
    frame_end = 0x55FA55FA

    # 读取帧头
    head_bytes = fid.read(4)
    if len(head_bytes) < 4:
        return None, None
    head_find = struct.unpack('<I', head_bytes)[0]  # 使用小端序
    if head_find != frame_head:
        return None, None

    # 读取帧长度
    length_bytes = fid.read(4)
    if len(length_bytes) < 4:
        return None, None
    frame_data_length = struct.unpack('<I', length_bytes)[0] * 4
    if frame_data_length <= 0 or frame_data_length > 1000000:
        return None, None

    # 检查帧尾
    current_pos = fid.tell()
    fid.seek(current_pos + frame_data_length - 12, 0)  # 偏移到结尾
    end_bytes = fid.read(4)
    if len(end_bytes) < 4:
        return None, None
    end_find = struct.unpack('<I', end_bytes)[0]
    if end_find != frame_end:
        return None, None

    # 回到文件开头
    fid.seek(-frame_data_length + 4, 1)

    # 读取参数
    # 读取前 3 个 uint32
    data_temp1_bytes = fid.read(12)
    if len(data_temp1_bytes) < 12:
        return None, None
    data_temp1 = np.frombuffer(data_temp1_bytes, dtype='<u4')  # 小端序uint32

    e_scan_az = data_temp1[1] * 0.01
    point_num_in_bowei = data_temp1[2]

    # point_num 合理性检查
    if point_num_in_bowei < 0 or point_num_in_bowei > 1000:
        print(f"点数异常: {point_num_in_bowei}")
        return None, None

    # 读取航迹信息和其他参数
    param_count = point_num_in_bowei * 4 + 5
    param_bytes = fid.read(param_count * 4)
    if len(param_bytes) < param_count * 4:
        return None, None
    data_temp = np.frombuffer(param_bytes, dtype='<u4')

    # 提取航迹信息
    if point_num_in_bowei > 0:
        track_no_info = data_temp[:point_num_in_bowei * 4]
    else:
        track_no_info = np.array([], dtype=np.uint32)

    # 提取其他参数
    base_idx = point_num_in_bowei * 4
    params = Parameters(
        e_scan_az=e_scan_az,
        track_no_info=track_no_info,
        freq=data_temp[base_idx] * 1e6,
        cpi_count=data_temp[base_idx + 1],
        prt_num=data_temp[base_idx + 2],
        prt=data_temp[base_idx + 3] * 0.0125e-6,
        data_length=data_temp[base_idx + 4]
    )

    # 参数验证
    if params.prt_num <= 0 or params.prt_num > 10000:
        logger.warning(f"PRT_num 异常: {params.prt_num}")
        return None, None
    if params.prt <= 0 or params.prt > 1:
        logger.warning(f"PRT 异常: {params.prt}")
        return None, None
    if params.freq <= 0 or params.freq > 1e12:
        logger.warning(f"频率异常: {params.freq}")
        return None, None

    # 读取 IQ 数据
    iq_data_len = params.prt_num * 31 * 2
    data_bytes = fid.read(iq_data_len * 4)
    if len(data_bytes) < iq_data_len * 4:
        print(f"IQ 数据长度不足: 期望{iq_data_len * 4}, 实际{len(data_bytes)}")
        return None, None

    data_out_temp = np.frombuffer(data_bytes, dtype='<f4')  # 小端序 float32

    # 重构复数数据
    data_out_real = data_out_temp[::2]
    data_out_imag = data_out_temp[1::2]
    data_out_complex = data_out_real + 1j * data_out_imag
    data_out = data_out_complex.reshape(31, params.prt_num, order='F')

    # 跳过帧尾
    fid.seek(4, 1)

    return params, data_out


def cfar_detector_2d(rd_matrix, cfar_threshold_factor,
                     detection_rows, detection_cols,
                     num_guard_cells_range, num_training_cells_doppler):
    """
    对RD矩阵执行2D-CFAR检测

    :param rd_matrix: 输入的 RD 矩阵 (linear)
    :param detection_rows: 距离维检测区域行索引范围
    :param detection_cols: 多普勒维检测区域列索引范围
    :param num_guard_cells_range: 距离维保护单元数 (单侧)
    :param num_training_cells_doppler: 多普勒维训练单元数 (单侧)
    :param cfar_threshold_factor: CFAR门限因子
    :return: 如果找到目标，返回一个应用了十字掩码的RD矩阵；否则返回None
    """

    if rd_matrix.size == 0:
        return None

    rows, cols = rd_matrix.shape

    zero_vel_idx = cols // 2

    # 排除零速度附近的杂波区域
    clutter_width = 3
    non_clutter_mask = np.ones(cols, dtype=bool)
    non_clutter_mask[max(0, zero_vel_idx - clutter_width):min(cols, zero_vel_idx + clutter_width + 1)] = False

    if not np.any(non_clutter_mask):
        return None

    # 调用修正的CFAR检测
    target_detected, target_mask, target_row, target_col = cfar(
        rd_matrix,
        detection_rows,
        detection_cols,
        num_guard_cells_range,
        num_training_cells_doppler,
        cfar_threshold_factor
    )

    if target_detected:
        result = rd_matrix * target_mask.astype(float)
        return result, target_row, target_col
    else:
        return None, None, None


def cfar(data_power, detection_rows, detection_cols, Gr, Td, threshold_factor):
    num_rows, num_cols = data_power.shape

    # 初始化
    local_detection_area = data_power[np.ix_(detection_rows, detection_cols)]
    local_num_rows, local_num_cols = local_detection_area.shape
    status_map = np.zeros((local_num_rows, local_num_cols), dtype=int)
    output_mask = np.zeros_like(data_power, dtype=bool)

    # 按幅度降序排序
    sorted_indices = np.argsort(local_detection_area.ravel())[::-1]
    rows_in_local, cols_in_local = np.unravel_index(sorted_indices, local_detection_area.shape)

    i, j = 0, 0
    for k in range(len(sorted_indices)):
        local_row = rows_in_local[k]
        local_col = cols_in_local[k]

        if status_map[local_row, local_col] != 0:
            continue

        # 转换为全局坐标
        i = detection_rows[local_row]
        j = detection_cols[local_col]
        cut_power = data_power[i, j]

        is_target = True

        # 距离向上方参考单元
        sum_noise_upper, count_noise_upper = 0, 0
        for r_ref in range(i - Gr - Td * 2, i - Gr):
            if 0 <= r_ref < num_rows:
                # 检查是否在检测区域内且已被处理
                if detection_rows[0] <= r_ref <= detection_rows[-1]:
                    local_r = r_ref - detection_rows[0]
                    if status_map[local_r, local_col] in [1, 2]:
                        continue
                sum_noise_upper += data_power[r_ref, j]
                count_noise_upper += 1

        if count_noise_upper == 0 or cut_power <= threshold_factor * (sum_noise_upper / count_noise_upper):
            is_target = False

        if not is_target:
            status_map[local_row, local_col] = 2
            continue

        # 距离向下方参考单元
        sum_noise_lower, count_noise_lower = 0, 0
        for r_ref in range(i + Gr + 1, i + Gr + Td * 2 + 1):
            if 0 <= r_ref < num_rows:
                if detection_rows[0] <= r_ref <= detection_rows[-1]:
                    local_r = r_ref - detection_rows[0]
                    if status_map[local_r, local_col] in [1, 2]:
                        continue
                sum_noise_lower += data_power[r_ref, j]
                count_noise_lower += 1

        if count_noise_lower == 0 or cut_power <= threshold_factor * (sum_noise_lower / count_noise_lower):
            is_target = False

        if not is_target:
            status_map[local_row, local_col] = 2
            continue

        # 最终判定
        if is_target:
            status_map[local_row, local_col] = 1
            output_mask[i, j] = True
        else:
            status_map[local_row, local_col] = 2

    return np.any(output_mask), output_mask, i, j


def process_batch(batch: BatchFile) -> Optional[Dict[str, Any]]:
    """
    处理批次文件
    :param batch: 批次文件
    :return: RD 图，距离轴，速度轴，识别率
    """
    fs = 20e6  # 采样率 (20 MHz)
    c = 3e8  # 光速 (m/s)
    delta_r = c / (2 * fs)  # 距离分辨率

    frame_count = 0
    rd_matrices = []
    ranges = []
    velocities = []
    point_index = []
    # 额外特征
    num_none_zero = []
    num_none_zero_row = []
    detected_rate = []
    total = 0
    detected = 0

    with open(batch.raw_file, 'rb') as fid:
        while True:
            params, data = read_raw_data(fid)
            if params is None:
                break

            frame_count += 1

            # 跳过没有航迹信息的帧
            if len(params.track_no_info) == 0:
                continue

            # 添加数据验证
            if len(params.track_no_info) < 4:
                continue

            # 验证参数有效性
            if params.prt <= 0 or params.prt_num <= 0 or params.freq <= 0:
                continue

            distance_bins, prt_bins = data.shape    # 距离单元数 (31) 和 PRT 单元数
            mtd_win = signal.windows.taylor(distance_bins, nbar=4, sll=30, norm=False).reshape(-1, 1)  # 生成泰勒窗
            coef_mtd_2d = np.repeat(mtd_win, prt_bins, axis=1)
            data_windowed = data * coef_mtd_2d      # 加窗处理
            mtd_result = fftshift(fft(data_windowed, axis=1), axes=1)   # FFT 处理

            # 计算多普勒速度轴
            delta_v = c / (2 * params.prt_num * params.prt * params.freq)
            if not np.isfinite(delta_v) or delta_v <= 0 or delta_v > 10000:
                logger.warning(f"警告：帧 {frame_count} delta_v异常: {delta_v}, 跳过该帧")
                continue

            half_prt = prt_bins // 2
            if half_prt <= 0 or half_prt > 10000:
                logger.warning(f"警告：帧 {frame_count} half_prt异常: {half_prt}, 跳过该帧")
                continue

            v_axis = np.linspace(-prt_bins / 2 * delta_v, prt_bins / 2 * delta_v, prt_bins, endpoint=False)
            if not np.all(np.isfinite(v_axis)) or len(v_axis) != params.prt_num:
                logger.warning(f"警告：帧 {frame_count} v_axis异常，长度:{len(v_axis)} != {params.prt_num}, 跳过该帧")
                continue

            # 目标检测
            amp_max_vr_unit = int(params.track_no_info[3])

            # 修正多普勒索引
            if amp_max_vr_unit > half_prt:
                amp_max_vr_unit = amp_max_vr_unit - half_prt
            else:
                amp_max_vr_unit = amp_max_vr_unit + half_prt

            amp_max_vr_unit = amp_max_vr_unit - 1   # 转换为 python 的 0-based 索引
            amp_max_vr_unit = np.clip(amp_max_vr_unit, 0, params.prt_num - 1)   # 确保索引在有效范围内

            # 目标中心位于第 16 个距离单元
            center_local_bin = 15
            local_radius = 10

            # 计算局部检测窗口
            range_start_local = max(0, center_local_bin - local_radius)
            range_end_local = min(mtd_result.shape[0], center_local_bin + local_radius + 1)
            doppler_start = max(0, amp_max_vr_unit - local_radius)
            doppler_end = min(mtd_result.shape[1], amp_max_vr_unit + local_radius + 1)

            target_sig = mtd_result[range_start_local:range_end_local, doppler_start:doppler_end]

            # 检测峰值
            abs_target = np.abs(target_sig)
            if abs_target.size == 0:
                continue

            max_idx = np.unravel_index(np.argmax(abs_target), abs_target.shape)
            amp_max_index_row, amp_max_index_col = max_idx

            # 获取目标全局距离单元索引
            global_range_bin = int(params.track_no_info[2])

            # 计算实际距离范围
            range_start_bin = global_range_bin - 15
            range_end_bin = global_range_bin + 15

            # 计算真实距离轴
            range_plot = np.arange(range_start_bin, range_end_bin + 1) * delta_r

            # 转换到全局距离位置
            detected_range_bin = range_start_local + amp_max_index_row
            if detected_range_bin >= len(range_plot):
                continue

            # 安全地计算多普勒速度
            doppler_idx = doppler_start + amp_max_index_col
            if doppler_idx >= len(v_axis):
                continue

            # 保存 MTD 处理结果
            rd_matrix = mtd_result
            range_axis = range_plot
            velocity_axis = v_axis.reshape(-1)
            rd_matrix = np.abs(rd_matrix)

            velocity_index = np.where(velocity_axis == 0)[0][0]
            point_df = pl.read_csv(batch.point_file, has_header=True, separator=",", encoding="gbk")
            index = min(params.track_no_info[1], len(point_df))
            doppler_velocity = point_df["多普勒速度"][int(index) - 1]
            if abs(doppler_velocity) > 5:
                rd_matrix[:, velocity_index - 1:velocity_index + 2] = 0

            # 2D-CFAR 检测和掩码生成
            processed_rd, target_row, target_col = cfar_detector_2d(
                rd_matrix=rd_matrix,
                detection_rows=np.arange(range_start_local, range_end_local),
                detection_cols=np.arange(doppler_start, doppler_end),
                num_guard_cells_range=3,
                num_training_cells_doppler=4,
                cfar_threshold_factor=5
            )
            total += 1
            if processed_rd is None:
                # logger.info(f"帧 {frame_count} 未找到目标，跳过该帧")
                continue
            detected += 1
            num_none_zero.append(np.count_nonzero(processed_rd))
            num_none_zero_row.append(np.count_nonzero(processed_rd[target_row - 1:target_row + 2, :]) / 3)
            detected_rate.append(detected / total)

            velocity_mask = np.abs(velocity_axis) < 56
            velocity_axis = velocity_axis[velocity_mask]
            rd_matrix = rd_matrix[:, velocity_mask]
            rd_matrix = np.clip(rd_matrix, 1, 1e10)
            rd_matrix = 20 * np.log10(rd_matrix)

            rd_matrices.append(rd_matrix[:, :, None])
            ranges.append(range_axis)
            velocities.append(velocity_axis)
            point_index.append(index)

    if rd_matrices is None:
        return None
    point_index = np.array(point_index, dtype=np.int32)
    point_index = point_index - point_index[0] + 1
    num_none_zero = np.array(num_none_zero, dtype=np.int32)
    num_none_zero_mean = np.cumsum(num_none_zero) / np.arange(1, len(num_none_zero) + 1)
    num_none_zero_row = np.array(num_none_zero_row, dtype=np.float32)
    num_none_zero_row_mean = np.cumsum(num_none_zero_row) / np.arange(1, len(num_none_zero_row) + 1)
    detected_rate = np.array(detected_rate, dtype=np.float32)
    extra_features = np.stack([num_none_zero_mean, num_none_zero_row_mean, detected_rate], axis=1).astype(np.float32)

    return {
        "rd_matrices": rd_matrices,
        "ranges": ranges,
        "velocities": velocities,
        "point_index": point_index,
        "extra_features": extra_features,
    }


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


if __name__ == '__main__':
    from tqdm import tqdm

    root = "/home/nju-student/mkh/datasets/radar"
    batch_files = get_batch_file_list(root)
    result = process_batch(batch_files[0])
    print(len(result["rd_matrices"]))
    # for batch_file in tqdm(batch_files):
    #     process_batch(batch_file)