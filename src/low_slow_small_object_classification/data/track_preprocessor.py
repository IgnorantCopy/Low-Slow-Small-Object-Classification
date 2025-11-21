import os
import re
import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial.distance import cosine

from src.low_slow_small_object_classification.data.utils import BatchFile

POINT_COLUMNS = ['时间', '批号', '距离', '方位', '俯仰', '多普勒速度', '和幅度', '信噪比', '原始点数量']
TRACK_COLUMNS = ['时间', '批号', '滤波距离', '滤波方位', '滤波俯仰', '全速度', 'X向速度', 'Y向速度', 'Z向速度', '航向']
FINAL_FEATURE_COLUMNS = [
    # 衍生特征
    "高度", "水平速度", "爬升/俯冲角度_弧度", "RCS",
    # 统计特征
    "最小全速度", "平均全速度", "平均水平速度", "平均高度",
    "最大高度", "最小高度", "高度波动范围", "高度标准差",
    "最大全速度", "最大水平速度", "最小水平速度", "水平速度波动范围",
    # 新增鸟类识别特征
    "转向角度累积均值", "转向角度累积标准差",
    "角速度累积均值", "角速度累积标准差",
    "多普勒抖动指数", "幅度抖动指数", "位置抖动指数",
    # 原始特征
    "俯仰", "多普勒速度", "和幅度", "信噪比", "原始点数量", "padding", "类别"
]


class DataLoader:
    @staticmethod
    def _load_point_data(point_file: str) -> pd.DataFrame:
        data = pd.read_csv(point_file, encoding='gbk', header=0, names=POINT_COLUMNS)
        data["时间"] = pd.to_datetime(data["时间"], format="%H:%M:%S.%f")
        return data

    @staticmethod
    def _load_track_data(track_file: str) -> pd.DataFrame:
        data = pd.read_csv(track_file, encoding='gbk', header=0, names=TRACK_COLUMNS)
        data["时间"] = pd.to_datetime(data["时间"], format="%H:%M:%S.%f")
        return data

    def load_data_pair(self, point_file: str, track_file: str):
        point_data = self._load_point_data(point_file)
        track_data = self._load_track_data(track_file)
        merged = pd.merge(point_data, track_data, on=['时间', '批号'], how='inner')
        merged = merged.sort_values('时间').reset_index(drop=True)
        merged["padding"] = np.zeros(len(merged))   # 填充列
        merged["类别"] = np.ones(len(merged)) * -1   # 类别列
        return merged


class OutlierCleaner:
    def __init__(self, outlier_threshold: float = 3.0, interpolation_method: str = 'linear',
                 velocity_threshold: float = 100.0, doppler_threshold: float = 50.0):
        self.outlier_threshold = outlier_threshold
        self.interpolation_method = interpolation_method
        self.velocity_threshold = velocity_threshold
        self.doppler_threshold = doppler_threshold

    def z_score(self, series: pd.Series, window_size: int = 10):
        outliers = pd.Series([False for _ in range(len(series))], index=series.index)
        for i in range(len(series)):
            start_index = max(0, i - window_size)
            window_series = series[start_index: i + 1]

            if len(window_series) > 3:
                mean = window_series.mean()
                std = window_series.std()
                if std > 1e-10:
                    z_score = abs((series[i] - mean) / std)
                    outliers[i] = z_score > self.outlier_threshold

        return outliers

    def iqr(self, series: pd.Series, window_size: int = 20) -> pd.Series:
        outliers = pd.Series([False for _ in range(len(series))], index=series.index)
        for i in range(len(series)):
            start_index = max(0, i - window_size)
            window_series = series[start_index: i + 1]

            if len(window_series) > 5:
                q1 = window_series.quantile(0.25)
                q3 = window_series.quantile(0.75)
                iqr = q3 - q1
                if iqr > 1e-10:
                    upper_bound = q3 + 1.5 * iqr
                    lower_bound = q1 - 1.5 * iqr
                    outliers[i] = (series[i] < lower_bound) or (series[i] > upper_bound)

        return outliers

    def extrapolate(self, series: pd.Series, outliers: pd.Series):
        if outliers.sum() == 0:
            return series.copy()

        series_copy = series.copy()
        indices = np.arange(len(series_copy))
        for i in indices[outliers]:
            prior_indices = indices[:i]
            prior_valid_mask = ~outliers[prior_indices]
            prior_valid_indices = prior_indices[prior_valid_mask]

            # 如果之前没有足够的有效点，尝试使用历史有效点的均值
            if len(prior_valid_indices) < 2:
                # 只使用当前点之前的有效数据
                prior_all_valid = indices[:i][~outliers[:i]]
                if len(prior_all_valid) >= 1:
                    fill_value = series_copy[prior_all_valid].mean()
                    if pd.isna(fill_value):
                        fill_value = 0
                    series_copy[i] = fill_value
                else:
                    series_copy[i] = 0
                    continue

            # 获取之前的有效点的数据
            x_prior = prior_valid_indices
            y_prior = series_copy[prior_valid_indices].values

            if self.interpolation_method == 'linear':
                model = np.polyfit(x_prior, y_prior, 1)
            elif self.interpolation_method == 'quadratic' and len(x_prior) >= 3:
                model = np.polyfit(x_prior, y_prior, 2)
            else:
                model = np.polyfit(x_prior, y_prior, 1)

            predicted_value = np.polyval(model, i)
            # 防止异常值
            if np.isnan(predicted_value) or np.isinf(predicted_value):
                predicted_value = series_copy[prior_valid_indices[-1]]

            series_copy[i] = predicted_value

        return series_copy

    def clean_data(self, data: pd.DataFrame):
        cleaned_data = data.copy()
        if len(cleaned_data) < 3:
            return cleaned_data

        # 重点处理多普勒速度
        doppler_series = cleaned_data['多普勒速度']
        z_score_outliers = self.z_score(doppler_series)
        iqr_outliers = self.iqr(doppler_series)
        combined_outliers = z_score_outliers | iqr_outliers | (np.abs(doppler_series) > self.doppler_threshold)
        cleaned_data['多普勒速度'] = self.extrapolate(doppler_series, combined_outliers)

        # 其他数值异常
        numeric_data = cleaned_data.select_dtypes(include=[np.number])
        for col in numeric_data.columns:
            if col in ["时间", "批号"]:
                continue

            series = numeric_data[col]
            outliers = self.z_score(series)
            cleaned_data[col] = self.extrapolate(series, outliers)

        return cleaned_data


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    df = pl.from_pandas(data)

    df = df.with_columns(
        (pl.col("滤波距离") * pl.col("滤波俯仰").radians().sin()).alias("高度"),
        (pl.col("X向速度").pow(2) + pl.col("Y向速度").pow(2)).sqrt().alias("水平速度"),
    )
    df = df.with_columns(
        pl.arctan2(pl.col("Z向速度"), pl.col("水平速度")).alias("爬升/俯冲角度_弧度"),
        (pl.col("和幅度") * pl.col("滤波距离").pow(4)).log10().alias("RCS"),
    )
    df = df.with_columns(
        # 当前时刻的速度向量模长
        (pl.col("X向速度").pow(2) + pl.col("Y向速度").pow(2) + pl.col("Z向速度").pow(2)).sqrt().alias(
            "三维速度模长"),
        # 计算时间差（用于角速度计算）
        (pl.col("时间").diff().dt.total_seconds()).alias("时间差"),
    )
    df = df.with_columns(
        # 相邻时刻速度向量点积
        (pl.col("X向速度") * pl.col("X向速度").shift(1) +
         pl.col("Y向速度") * pl.col("Y向速度").shift(1) +
         pl.col("Z向速度") * pl.col("Z向速度").shift(1)).alias("速度向量点积"),
        # 相邻时刻速度向量模长乘积
        (pl.col("三维速度模长") * pl.col("三维速度模长").shift(1)).alias("速度模长乘积"),
    )
    df = df.with_columns(
        # 转向角度 = arccos(dot_product / (|v1| * |v2|))
        pl.when(pl.col("速度模长乘积") > 1e-6)
        .then((pl.col("速度向量点积") / pl.col("速度模长乘积")).clip(-1.0, 1.0).arccos())
        .otherwise(0.0).alias("转向角度"),
    )
    df = df.with_columns(
        # 角速度 = 转向角度 / 时间差
        pl.when(pl.col("时间差") > 1e-6)
        .then(pl.col("转向角度") / pl.col("时间差"))
        .otherwise(0.0).alias("角速度"),
    )
    df = df.with_columns(
        # 多普勒速度的二阶差分
        pl.col("多普勒速度").diff().diff().alias("多普勒二阶差分"),
        # 幅度的二阶差分
        pl.col("和幅度").diff().diff().alias("幅度二阶差分"),
        # 位置的二阶差分
        pl.col("滤波距离").diff().diff().alias("距离二阶差分"),
        pl.col("滤波方位").diff().diff().alias("方位二阶差分"),
        pl.col("滤波俯仰").diff().diff().alias("俯仰二阶差分"),
    )
    df = df.with_columns(
        # 原有特征
        pl.col("全速度").cum_min().alias("最小全速度"),
        (pl.col("全速度").cum_sum() / pl.col("全速度").cum_count()).alias("平均全速度"),
        (pl.col("水平速度").cum_sum() / pl.col("水平速度").cum_count()).alias("平均水平速度"),
        (pl.col("高度").cum_sum() / pl.col("高度").cum_count()).alias("平均高度"),
        pl.col("高度").cum_max().alias("最大高度"),
        pl.col("高度").cum_min().alias("最小高度"),
        pl.col("全速度").cum_max().alias("最大全速度"),
        pl.col("水平速度").cum_max().alias("最大水平速度"),
        pl.col("水平速度").cum_min().alias("最小水平速度"),
        # 新增：转向角度和角速度的累积统计
        (pl.col("转向角度").cum_sum() / pl.col("转向角度").cum_count()).alias("转向角度累积均值"),
        (pl.col("角速度").cum_sum() / pl.col("角速度").cum_count()).alias("角速度累积均值"),
    )
    df = df.with_columns(
        (pl.col("高度").cum_max() - pl.col("高度").cum_min()).alias("高度波动范围"),
        (pl.col("水平速度").cum_max() - pl.col("水平速度").cum_min()).alias("水平速度波动范围"),
        # 累积标准差（使用正确的方差公式）
        (pl.col("高度").pow(2).cum_sum() / pl.col("高度").cum_count() -
         (pl.col("高度").cum_sum() / pl.col("高度").cum_count()).pow(2)).sqrt().alias("高度标准差"),
        # 转向角度和角速度的累积标准差
        (pl.col("转向角度").pow(2).cum_sum() / pl.col("转向角度").cum_count() -
         (pl.col("转向角度").cum_sum() / pl.col("转向角度").cum_count()).pow(2)).sqrt().alias("转向角度累积标准差"),
        (pl.col("角速度").pow(2).cum_sum() / pl.col("角速度").cum_count() -
         (pl.col("角速度").cum_sum() / pl.col("角速度").cum_count()).pow(2)).sqrt().alias("角速度累积标准差"),
        # 抖动指数（二阶差分的RMS）
        (pl.col("多普勒二阶差分").pow(2).cum_sum() / pl.col("多普勒二阶差分").cum_count()).sqrt().alias(
            "多普勒抖动指数"),
        (pl.col("幅度二阶差分").pow(2).cum_sum() / pl.col("幅度二阶差分").cum_count()).sqrt().alias("幅度抖动指数"),
        ((pl.col("距离二阶差分").pow(2) + pl.col("方位二阶差分").pow(2) + pl.col("俯仰二阶差分").pow(2))
         .cum_sum() / pl.col("距离二阶差分").cum_count()).sqrt().alias("位置抖动指数"),
    )

    # 选择最终特征
    df_final_features = df.select(FINAL_FEATURE_COLUMNS)

    # 填充所有因计算差分等产生的空值
    df_final_features = df_final_features.fill_null(0.0).fill_nan(0.0)

    return df_final_features.to_pandas()


class TrajectoryPreprocessor:
    def __init__(self, seq_len: int, test: bool = False):
        self.seq_len = seq_len
        self.test = test

        self.outlier_cleaner = OutlierCleaner()

    def process_single_trajectory(self, batch_file: BatchFile):
        point_file = batch_file.point_file
        track_file = batch_file.track_file

        # 从文件名提取信息
        pattern = r"PointTracks_(\d+)_(\d+)\.txt" if self.test else r"PointTracks_(\d+)_(\d+)_(\d+)\.txt"
        re_result = re.match(pattern, os.path.basename(point_file))
        if not re_result:
            return None

        batch_id = re_result.group(1)
        if self.test:
            label = 0
        else:
            label = int(re_result.group(2))
            # 只处理 1-4 类标签
            if label > 4:
                return None

        num_points = int(re_result.group(2 if self.test else 3))

        merged_data = DataLoader().load_data_pair(point_file, track_file)
        if merged_data is None or len(merged_data) == 0:
            return None

        cleaned_data = self.outlier_cleaner.clean_data(merged_data)
        features = feature_engineering(cleaned_data)
        sequence = self._create_sequence(features)
        if sequence is None:
            return None

        return {
            "sequence": sequence,
            "label": label - 1,
            "num_points": num_points,
            "batch_id": batch_id,
        }

    @staticmethod
    def data_padding(merged_data, track_seq_len, n=4):
        """
        对点迹数据进行智能填充，通过寻找历史相似模式
        :param merged_data: 点迹数据 (NumPy Array)
        :param track_seq_len: 目标长度
        :param n: 用于模式匹配的行数,不超过 6
        :return: 填充后的完整数据
        """
        current_len = merged_data.shape[0]
        padding_length = track_seq_len - current_len
        reference_pattern = merged_data[-n:, :]  # shape: [n, features]

        # 确定搜索范围
        if current_len >= padding_length + 2 * n:
            search_end = min(current_len - padding_length - n, current_len - 2 * n)
        else:
            search_end = current_len - 2 * n
        search_end = max(0, search_end)

        min_distance = float("inf")
        best_start_idx = n + 1

        # 遍历可能的起始位置

        for start_idx in range(search_end):
            if start_idx + n > current_len:
                break
            current_pattern = merged_data[
                start_idx: start_idx + n, :
            ]  # shape: [n, features]
            total_distance = 0.0
            for i in range(n):
                row_distance_uc = np.sqrt(
                    np.sum((reference_pattern[i] - current_pattern[i]) ** 2)
                )
                row_distance_cos = cosine(reference_pattern[i], current_pattern[i])
                row_distance = (row_distance_uc + row_distance_cos) / 2
                total_distance += row_distance
            if total_distance < min_distance:
                min_distance = total_distance
                best_start_idx = start_idx + n + 1

        padding_data_list = []
        remaining_padding = padding_length
        while remaining_padding > 0:
            # 确定这次可以填充多少行
            available_rows = current_len - best_start_idx
            rows_to_add = min(remaining_padding, available_rows)

            padding_segment = merged_data[best_start_idx: best_start_idx + rows_to_add, :]
            padding_data_list.append(padding_segment)
            remaining_padding -= rows_to_add

            # 如果还需要更多数据，重新开始循环
            if remaining_padding > 0 and available_rows < remaining_padding:
                pass

        # 合并所有填充数据
        if padding_data_list:
            padding_data = np.concatenate(padding_data_list, axis=0)
            assert (
                    padding_data.shape[0] == padding_length
            ), f"填充数据行数不匹配: {padding_data.shape[0]}, 期望: {padding_length}"
        else:
            # 如果没有找到合适的模式，回退到重复最后一行
            padding_data = np.stack(
                [merged_data[-1, :] for _ in range(padding_length)], axis=0
            )
        padding_data[:, -2] = 1
        return padding_data

    def _create_sequence(self, features: pd.DataFrame):
        if len(features) == 0:
            return None
        feature_array = features.to_numpy(dtype=np.float32)

        if len(feature_array) >= self.seq_len:
            sequence = feature_array[:self.seq_len]
        else:
            sequence = np.zeros((self.seq_len, feature_array.shape[1]), dtype=np.float32)
            sequence[:len(feature_array)] = feature_array

            sequence[len(feature_array):, :] = self.data_padding(feature_array, self.seq_len, n=4)

        return sequence
