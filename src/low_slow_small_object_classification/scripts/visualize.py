import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def visualize_rd_matrix(rd_matrix, range_axis, velocity_axis, batch, label, frame, save_path=None, verbose=False):
    """可视化 RD 矩阵"""
    if verbose:
        print(f"RD矩阵形状: {rd_matrix.shape}")

    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(velocity_axis, range_axis)
    plt.pcolormesh(X, Y, rd_matrix, shading='auto', cmap='jet')
    plt.xlim(-30, 30)
    plt.colorbar(label='Amplitude (dB)')
    plt.xlabel('velocity (m/s)')
    plt.ylabel('range (m)')
    plt.title(f'RD Map (Batch_{batch}_Label_{label}_Frame_{frame})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_3d_trajectory(point_file, save_path=None):
    # 绘制归一化距离的3D轨迹
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    df = pd.read_csv(point_file, header=None, encoding="gbk", skiprows=1)
    df.columns = [
        "time",
        "track_id",
        "range",
        "azimuth",
        "elevation",
        "doppler",
        "amplitude",
        "snr",
        "point_count",
    ]

    # 添加文件信息列，便于后续区分数据来源
    file_parts = os.path.basename(point_file).split("_")
    file_id = file_parts[1]  # 提取文件编号
    label = file_parts[2]  # 提取标签信息

    df["file_id"] = file_id
    df["label"] = label

    # 解析时间为 datetime
    df["timestamp"] = pd.to_datetime(df["time"], format="%H:%M:%S.%f")

    # 对每个文件的时间戳进行归一化处理
    min_times = df.groupby("file_id")["timestamp"].min()
    df["normalized_time"] = df.apply(
        lambda row: (row["timestamp"] - min_times[row["file_id"]]).total_seconds(), axis=1
    )

    # 对每个轨迹的距离进行归一化处理，使它们都从同一起点出发
    # 计算每个轨迹的初始距离
    initial_ranges = df.groupby(["track_id"])["range"].first()
    # 创建归一化距离列，表示相对于初始位置的距离变化
    df["normalized_range"] = df.apply(
        lambda row: row["range"] - initial_ranges[row["track_id"]], axis=1
    )

    for track_id, group in df.groupby("track_id"):
        # 获取该轨迹的标签
        label = group["label"].iloc[0]

        # 使用归一化的距离但保留原始角度
        x = (
            group["normalized_range"]
            * np.sin(np.radians(group["azimuth"]))
            * np.cos(np.radians(group["elevation"]))
        )
        y = (
            group["normalized_range"]
            * np.cos(np.radians(group["azimuth"]))
            * np.cos(np.radians(group["elevation"]))
        )
        z = group["normalized_range"] * np.sin(np.radians(group["elevation"]))

        ax.plot(x, y, z, label=f"Track {track_id} (Label {label})")

    ax.set_xlabel("X (relative)")
    ax.set_ylabel("Y (relative)")
    ax.set_zlabel("Z (relative)")
    ax.set_title("3D Target Trajectory (Normalized Range)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(conf_matrix: np.ndarray, class_names=None, title='Confusion Matrix',
                          normalize=False, width=800, height=600):
    # 如果需要归一化
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix = np.round(conf_matrix * 100, 2)  # 转换为百分比并保留两位小数
        z_text = [[f'{val}%' for val in row] for row in conf_matrix]
        title = title + ' (Normalized, %)'
    else:
        z_text = [[f'{val}' for val in row] for row in conf_matrix]

    # 如果没有提供类别名称，则使用数字作为标签
    if class_names is None:
        class_names = [str(i) for i in range(conf_matrix.shape[0])]

    # 创建热力图数据
    fig = ff.create_annotated_heatmap(
        z=conf_matrix,
        x=class_names,
        y=class_names,
        annotation_text=z_text,
        colorscale=px.colors.sequential.Blues,
        showscale=True,
        hoverinfo='z'
    )

    # 更新布局
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20),
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis=dict(
            title=dict(text='Predicted Labels', font=dict(size=16)),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(text='True Labels', font=dict(size=16)),
            tickfont=dict(size=12)
        ),
        width=width,
        height=height,
        margin=dict(l=80, r=80, t=100, b=80)
    )

    # 添加悬停信息
    fig.update_traces(
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Value: %{z}<extra></extra>'
    )

    return fig
