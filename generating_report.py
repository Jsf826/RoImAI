import json
import os
import traceback
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from matplotlib import font_manager
import warnings
import json
import numpy as np
from shapely.geometry import Polygon, box
from collections import Counter
import csv
import pandas as pd
import xlsxwriter
import scipy.stats as stats
from itertools import combinations
import math
from shapely.geometry import Polygon, Point, LineString
from shapely.strtree import STRtree
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

# 全局配置
warnings.filterwarnings("ignore", category=RuntimeWarning, module="matplotlib")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False

# 粒径区间定义
GRAIN_SIZE_INTERVALS_10 = [
    (64, 256), (4, 64), (2, 4), (0.5, 2),
    (0.25, 0.5), (0.125, 0.25), (0.0625, 0.125),
    (0.03125, 0.0625), (0.0156, 0.03125), (0, 0.0156)
]
GRAIN_SIZE_INTERVALS_12 = [
    (-float('inf'),-8),(-8.0, -6.0), (-6.0, -2.0), (-2.0, -1.0), (-1.0, 0.0),
    (0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0),
    (4.0, 5.0), (5.0, 8.0), (8.0, float('inf'))
]
GRAIN_SIZE_INTERVALS_32 = [
    (-float('inf'), -8.0), (-8.0, -6.0), (-6.0, -2.0), (-2.0, -1.0),
    (-1.0, -.75), (-.75, -.5), (-.5, -.25), (-25.0, 0), (0, 0.25),
    (0.25, 0.5), (.5, 0.75), (0.75, 1), (1, 1.25), (1.25, 1.5),
    (1.5, 1.75), (1.75, 2), (2, 2.25), (2.25, 2.5), (2.5, 2.75),
    (2.75, 3), (3, 3.25), (3.25, 3.5), (3.5, 3.75), (3.75, 4),
    (4, 4.25), (4.25, 4.5), (4.5, 4.75), (4.75, 5), (5, 6),
    (6, 7), (7, 8), (8.0, float('inf'))
]

ROUNDNESS_INTERVALS = [
    (0.0, 0.5),   # 棱角
    (0.5, 0.7),  # 次棱
    (0.7, 0.8),  # 次圆
    (0.8, 0.85),  # 圆
    (0.85, 1.00)   # 极圆
]

#ROUNDNESS_LABELS = ["棱角", "次棱", "次圆", "圆", "极圆"]
ROUNDNESS_LABELS = ["Angular", "Sub angular", "Sub rounded", "Rounded", "Highly rounded"]

# ["巨砾", "粗砾", "中砾", "细砾", "巨砂", "粗砂", "中砂", "细砂", "极细砂", "粗粉砂", "细粉砂", "泥"]
GRAIN_CLASSES_12 = ["Giant gravel", "Coarse gravel", "Medium gravel", "Fine gravel", "Giant sand",
                    "Coarse sand", "Medium sand", "Fine sand", "Very fine sand", "Coarse silt", "Fine silt", "Clay"]


def load_data(file_path):
    """
    从指定路径加载 JSON 数据。
    :param file_path: JSON 文件路径
    :return: 解析后的 JSON 数据（字典格式）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def calculate_area(points):
    """
    根据多边形的点坐标计算其面积。
    :param points: 多边形的点坐标列表 [(x1, y1), (x2, y2), ...]
    :return: 多边形的面积
    """
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n  # 下一个点的索引（循环）
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0

def calculate_perimeter(points):
    perimeter = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        dx = points[i][0] - points[j][0]
        dy = points[i][1] - points[j][1]
        perimeter += math.sqrt(dx ** 2 + dy ** 2)
    return perimeter

def calculate_bounding_boxes(shapes):
    """从每个形状的点集合计算外接矩形框"""
    bounding_boxes = []

    for shape in shapes:
        points = shape['points']
        x_coords, y_coords = zip(*points)  # 分别获取 x 和 y 坐标
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        bounding_box = (x_min, y_min, x_max, y_max)
        bounding_boxes.append(bounding_box)

    return bounding_boxes

def calculate_iou(bbox1, bbox2):
    """计算两个外接矩形框的 IoU 值"""
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2

    # 计算交集区域
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # 计算每个框的面积
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)

    # 计算 IoU
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

# 绘制饼图
def plot_pie_chart(data, labels, title, output_path):
    # 过滤掉为0的类别和对应数据
    filtered_data_labels = [(d, l) for d, l in zip(data, labels) if d > 0]
    filtered_data, filtered_labels = zip(*filtered_data_labels) if filtered_data_labels else ([], [])

    # 检查是否还有数据可绘制
    if not filtered_data:
        print("No data to plot.")
        return


# 1. 碎屑颗粒粒径分析
def calculate_longest_diameter(points, scale=1):
    """计算颗粒的最大直径"""
    max_distance = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = math.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
            max_distance = max(max_distance, distance / scale)
    return max_distance


def calculate_main_grain_range(data, scale):
    """
    计算主要粒径区间和最大粒径
    """
    # 定义10个粒径区间
    GRAIN_SIZE_INTERVALS_10 = [
        (64, 256), (4, 64), (2, 4), (0.5, 2),
        (0.25, 0.5), (0.125, 0.25), (0.0625, 0.125),
        (0.03125, 0.0625), (0.0156, 0.03125), (0, 0.0156)
    ]

    # 初始化累积面积字典和变量
    interval_areas = {str(interval): 0 for interval in GRAIN_SIZE_INTERVALS_10}
    total_area = 0.0
    max_diameter = 0.0  # 初始化最大粒径

    # 遍历所有颗粒，计算其直径、面积并分配到粒径区间
    for shape in data.get('shapes', []):
        points = shape['points']
        diameter = calculate_longest_diameter(points, scale)  # 计算最长直径
        area = calculate_area(points)  # 计算颗粒面积
        total_area += area
        max_diameter = max(max_diameter, diameter)  # 更新最大粒径

        # 分配到对应的粒径区间
        for interval in GRAIN_SIZE_INTERVALS_10:
            min_d, max_d = interval
            if min_d <= diameter < max_d:
                interval_areas[str(interval)] += area
                break

    # 计算每个区间的面积占比
    interval_proportions = {k: (v / total_area) * 100 for k, v in interval_areas.items()}

    # 计算累积面积百分比
    cumulative_area_percentage = 0
    cumulative_proportions = {}
    for interval, proportion in interval_proportions.items():
        cumulative_area_percentage += proportion
        cumulative_proportions[interval] = cumulative_area_percentage

    # 找到累积面积占比在 25% 至 75% 的粒径区间
    selected_intervals = [
        interval for interval, proportion in cumulative_proportions.items()
        if 25 <= proportion <= 75
    ]

    # 确定主要粒径区间
    if len(selected_intervals) == 1:
        main_grain_range = selected_intervals[0]
    elif len(selected_intervals) == 2:
        # 检查是否相邻
        indices = [GRAIN_SIZE_INTERVALS_10.index(eval(interval)) for interval in selected_intervals]
        if abs(indices[0] - indices[1]) == 1:
            main_grain_range = f"{selected_intervals[0]} + {selected_intervals[1]}"
        else:
            main_grain_range = max(selected_intervals, key=lambda x: interval_proportions[x])
    elif len(selected_intervals) == 3:
        # 检查是否有相邻的两个区间
        indices = [GRAIN_SIZE_INTERVALS_10.index(eval(interval)) for interval in selected_intervals]
        if abs(indices[0] - indices[1]) == 1 or abs(indices[1] - indices[2]) == 1:
            main_grain_range = f"{selected_intervals[1]} + {selected_intervals[2]}"
        else:
            main_grain_range = max(selected_intervals, key=lambda x: interval_proportions[x])
    else:
        main_grain_range = "Can't find main grain range."

    return max_diameter, main_grain_range, interval_proportions
# 2. 组分含量分析
def component_content_analysis(data, scale, output_dir):
    """
    组分含量分析，分为一级分类和二级分类。
    输出为绝对含量和相对含量，并保存到 CSV 文件和饼图。
    """
    # # 定义二级分类和一级分类的对应关系
    # second_level_labels = {
    #     "石英类-石英": "石英",
    #     "石英类-燧石": "石英",
    #     "长石-斜长石": "长石",
    #     "长石-钾长石": "长石",
    #     "岩屑-沉积岩": "岩屑",
    #     "岩屑-火成岩": "岩屑",
    #     "岩屑-变质岩": "岩屑",
    #     "岩屑-火山碎屑": "岩屑",
    #     "岩屑-其他矿物碎屑": "岩屑"
    # }
    # first_level_labels = ["石英", "长石", "岩屑"]

    second_level_labels = {
        "石英类-石英": "quartz",
        "石英类-燧石": "quartz",
        "长石-斜长石": "feldspar",
        "长石-钾长石": "feldspar",
        "岩屑-沉积岩": "rock_fragment",
        "岩屑-火成岩": "rock_fragment",
        "岩屑-变质岩": "rock_fragment",
        "岩屑-火山碎屑": "rock_fragment",
        "岩屑-其他矿物碎屑": "rock_fragment",
    }
    first_level_labels = ["quartz", "feldspar", "rock_fragment"]

    # 二级分类中英文对照标签
    second_level_labels_mapping = {
        "石英类-石英": "quartz",
        "石英类-燧石": "flint",
        "长石-斜长石": "plagioclase",
        "长石-钾长石": "orthoclase",
        "岩屑-沉积岩": "sedimentary_rock",
        "岩屑-火成岩": "igneous_rock",
        "岩屑-变质岩": "metamorphic_rock",
        "岩屑-火山碎屑": "volcanic_clastic",
        "岩屑-其他矿物碎屑": "other_mineral_clastic",
    }

    # 初始化面积字典
    second_level_areas = {label: 0.0 for label in second_level_labels.keys()}
    first_level_areas = {label: 0.0 for label in first_level_labels}

    # 总图像面积（用于计算绝对含量）
    total_image_area = 0.0
    width = data.get('imageWidth', 0)
    height = data.get('imageHeight', 0)
    if width > 0 and height > 0:
        total_image_area = width * height / (scale ** 2)

    # 遍历数据，计算各组分的面积
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        points = shape.get("points", [])
        area = calculate_area(points) / (scale ** 2)  # 按照比例尺计算面积

        if label in second_level_labels_mapping:
            second_level_areas[label] += area
            first_level_label = second_level_labels[label]
            first_level_areas[first_level_label] += area

    # 初始化英文二级标签面积字典
    second_level_areas_english = {label: 0.0 for label in second_level_labels_mapping.values()}

    # 填充英文二级标签面积字典
    for chinese_label, area in second_level_areas.items():
        if chinese_label in second_level_labels_mapping:
            english_label = second_level_labels_mapping[chinese_label]
            second_level_areas_english[english_label] += area

    # 计算绝对含量
    absolute_proportions_second = {
        label: (area / total_image_area) * 100 for label, area in second_level_areas_english.items() if
        total_image_area > 0
    }
    absolute_proportions_first = {
        label: (area / total_image_area) * 100 for label, area in first_level_areas.items() if total_image_area > 0
    }

    # 计算相对含量
    total_second_level_area = sum(second_level_areas_english.values())  # 基于英文标签计算总面积
    relative_proportions_second = {
        label: (area / total_second_level_area) * 100 if total_second_level_area > 0 else 0
        for label, area in second_level_areas_english.items()
    }

    total_first_level_area = sum(first_level_areas.values())
    relative_proportions_first = {
        label: (area / total_first_level_area) * 100 if total_first_level_area > 0 else 0
        for label, area in first_level_areas.items()
    }

    # 保存为 CSV 文件
    second_level_csv_path = os.path.join(output_dir, "second_level_composition.csv")
    first_level_csv_path = os.path.join(output_dir, "first_level_composition.csv")

    pd.DataFrame({
        "secondary_classification": list(absolute_proportions_second.keys()),
        "absolute_proportions": list(absolute_proportions_second.values()),
        "relative_proportions": list(relative_proportions_second.values())
    }).to_csv(second_level_csv_path, index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "first_classification": list(absolute_proportions_first.keys()),
        "absolute_proportion": list(absolute_proportions_first.values()),
        "relative_proportion": list(relative_proportions_first.values())
    }).to_csv(first_level_csv_path, index=False, encoding="utf-8-sig")

    print(f"二级分类组分含量分析已保存到: {second_level_csv_path}")
    print(f"一级分类组分含量分析已保存到: {first_level_csv_path}")

    # 绘制饼图
    plot_pie_chart(
        list(absolute_proportions_second.values()),
        list(absolute_proportions_second.keys()),
        "second_level_composition",
        os.path.join(output_dir, "second_level_composition.png")
    )

    plot_pie_chart(
        list(absolute_proportions_first.values()),
        list(absolute_proportions_first.keys()),
        "first_level_composition",
        os.path.join(output_dir, "first_level_composition.png")
    )

    # 返回结果以供后续分析使用
    return {
        "absolute_proportions_second": absolute_proportions_second,
        "relative_proportions_second": relative_proportions_second,
        "absolute_proportions_first": absolute_proportions_first,
        "relative_proportions_first": relative_proportions_first
    }

    # # 计算绝对含量
    # absolute_proportions_second = {
    #     label: (area / total_image_area) * 100 for label, area in second_level_areas.items() if total_image_area > 0
    # }
    # absolute_proportions_first = {
    #     label: (area / total_image_area) * 100 for label, area in first_level_areas.items() if total_image_area > 0
    # }
    #
    # # 计算相对含量
    # total_second_level_area = sum(second_level_areas.values())
    # relative_proportions_second = {
    #     label: (area / total_second_level_area) * 100 if total_second_level_area > 0 else 0
    #     for label, area in second_level_areas.items()
    # }
    #
    # total_first_level_area = sum(first_level_areas.values())
    # relative_proportions_first = {
    #     label: (area / total_first_level_area) * 100 if total_first_level_area > 0 else 0
    #     for label, area in first_level_areas.items()
    # }
    #
    # # 保存为 CSV 文件
    # second_level_csv_path = os.path.join(output_dir, "second_level_composition.csv")
    # first_level_csv_path = os.path.join(output_dir, "first_level_composition.csv")
    #
    # pd.DataFrame({
    #     "secondary_classification": list(absolute_proportions_second.keys()),
    #     "absolute_proportions": list(absolute_proportions_second.values()),
    #     "relative_proportions": list(relative_proportions_second.values())
    # }).to_csv(second_level_csv_path, index=False, encoding="utf-8-sig")
    #
    # pd.DataFrame({
    #     "first_classification": list(absolute_proportions_first.keys()),
    #     "absolute_proportion": list(absolute_proportions_first.values()),
    #     "relative_proportion": list(relative_proportions_first.values())
    # }).to_csv(first_level_csv_path, index=False, encoding="utf-8-sig")
    #
    # print(f"二级分类组分含量分析已保存到: {second_level_csv_path}")
    # print(f"一级分类组分含量分析已保存到: {first_level_csv_path}")
    #
    # # 绘制饼图
    # plot_pie_chart(
    #     list(absolute_proportions_second.values()),
    #     list(absolute_proportions_second.keys()),
    #     "second_level_composition",
    #     os.path.join(output_dir, "second_level_composition.png")
    # )
    #
    # plot_pie_chart(
    #     list(absolute_proportions_first.values()),
    #     list(absolute_proportions_first.keys()),
    #     "first_level_composition",
    #     os.path.join(output_dir, "first_level_composition.png")
    # )
    #
    # # 返回结果以供后续分析使用
    # return {
    #     "absolute_proportions_second": absolute_proportions_second,
    #     "relative_proportions_second": relative_proportions_second,
    #     "absolute_proportions_first": absolute_proportions_first,
    #     "relative_proportions_first": relative_proportions_first
    # }


# 3. 粒度分析结果
def grain_size_analysis(data, scale, estimated_basal_content, output_dir):
    # 计算每个颗粒的真实面积和最大直径
    diameters = []
    areas = []

    for shape in data.get('shapes', []):
        points = shape['points']
        diameter = calculate_longest_diameter(points, scale)
        area = calculate_area(points) / (scale ** 2)  # 将面积转换为实际单位
        diameters.append(diameter)
        areas.append(area)

    # 计算φ值
    phi_values = [0.3815 + 0.9027 * (-math.log2(d)) for d in diameters if d > 0]

    # 剔除φ值大于5的颗粒
    valid_phi_values = [phi for phi, area in zip(phi_values, areas) if phi <= 5]

    # 初始化32个区间的分布统计
    grain_distribution_32 = {f"({start}, {end})": 0 for start, end in GRAIN_SIZE_INTERVALS_32}

    # 统计每个φ值区间的颗粒面积
    for phi, area in zip(phi_values, areas):
        for start, end in GRAIN_SIZE_INTERVALS_32:
            if start <= phi < end:
                grain_distribution_32[f"({start}, {end})"] += area
                break

    # 总面积
    total_area = sum(grain_distribution_32.values())

    # 计算32个区间的面积频率和累积频率
    area_frequencies_32 = {k: (v / total_area) * 100 for k, v in grain_distribution_32.items()}
    cumulative_frequencies_32 = np.cumsum(list(area_frequencies_32.values()))

    # 初始化12个区间的分布统计
    grain_distribution_12 = {f"({start}, {end})": 0 for start, end in GRAIN_SIZE_INTERVALS_12}

    # 汇总32个区间的频率到12个区间
    for key, freq in area_frequencies_32.items():
        start_32, end_32 = map(float, key.strip("()").split(", "))
        for start_12, end_12 in GRAIN_SIZE_INTERVALS_12:
            if start_12 <= start_32 < end_12:
                grain_distribution_12[f"({start_12}, {end_12})"] += freq

    # 计算12区间的面积频率和累积频率
    area_frequencies_12 = list(grain_distribution_12.values())
    cumulative_frequencies_12 = np.cumsum(area_frequencies_12)

    # 获取特定百分位的φ值
    percentiles = [5, 16, 25, 50, 75, 84, 95]
    phi_percentiles = np.percentile(valid_phi_values, percentiles)

    phi_5, phi_16, phi_25, phi_50, phi_75, phi_84, phi_95 = phi_percentiles

    # 图解法参数计算
    Mz = (phi_16 + phi_50 + phi_84) / 3
    sigma_1 = (phi_84 - phi_16) / 4 + (phi_95 - phi_5) / 6.6
    Sk1 = ((phi_16 + phi_84 - 2 * phi_50) / (2 * (phi_84 - phi_16))) + ((phi_5 + phi_95 - 2 * phi_50) / (2 * (phi_95 - phi_5)))
    Kg = (phi_95 - phi_5) / (2.44 * (phi_75 - phi_25))

    # C值和M值计算
    C_phi = np.percentile(valid_phi_values, 1)
    M_phi = np.percentile(valid_phi_values, 50)
    C_value = 2 ** (-(C_phi - 0.3815) / 0.9027)
    M_value = 2 ** (-(M_phi - 0.3815) / 0.9027)

    # 矩法计算
    x_i = [(start + end) / 2 for start, end in GRAIN_SIZE_INTERVALS_12]
    x_i[0], x_i[-1] = -9, 9  # 处理边界
    f_i = area_frequencies_12
    Md = sum(x * f / 100 for x, f in zip(x_i, f_i))
    sigma = math.sqrt(sum(((x - Md) ** 2) * f / 100 for x, f in zip(x_i, f_i)))
    Sk = sum(((x - Md) ** 3) * f / 100 for x, f in zip(x_i, f_i)) / sigma ** 3
    K = sum(((x - Md) ** 4) * f / 100 for x, f in zip(x_i, f_i)) / sigma ** 4

    # 保存频率结果
    output_path = os.path.join(output_dir, "粒度分析结果.csv")
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Interval", "Frequency(%)", "Cumulate Frequency(%)"])
        for (start, end), freq, cum_freq in zip(GRAIN_SIZE_INTERVALS_12, area_frequencies_12, cumulative_frequencies_12):
            writer.writerow([f"({start}, {end})", round(freq, 3), round(cum_freq, 3)])

        writer.writerow([])
        writer.writerow(["32 intervals frequency and cumulative frequency"])
        for (start, end), freq, cum_freq in zip(GRAIN_SIZE_INTERVALS_32, area_frequencies_32.values(), cumulative_frequencies_32):
            writer.writerow([f"({start}, {end})", round(freq, 3), round(cum_freq, 3)])

        writer.writerow([])
        writer.writerow(["图解法参数"])
        writer.writerow(["平均粒径 (Mz)", round(Mz, 3)])
        writer.writerow(["标准偏差 (σ1)", round(sigma_1, 3)])
        writer.writerow(["偏度 (Sk1)", round(Sk1, 3)])
        writer.writerow(["峰度 (Kg)", round(Kg, 3)])
        writer.writerow(["C值", round(C_value, 3)])
        writer.writerow(["M值", round(M_value, 3)])

        writer.writerow([])
        writer.writerow(["矩法参数"])
        writer.writerow(["平均粒径 (Md)", round(Md, 3)])
        writer.writerow(["标准偏差 (σ)", round(sigma, 3)])
        writer.writerow(["偏度 (Sk)", round(Sk, 3)])
        writer.writerow(["峰度 (K)", round(K, 3)])

    print(f"粒度分析结果已保存到 {output_path}")

    return {
        "phi_values": valid_phi_values,
        "area_frequencies_12": area_frequencies_12,
        "cumulative_frequencies_12": cumulative_frequencies_12,
        "area_frequencies_32": area_frequencies_32,
        "cumulative_frequencies_32": cumulative_frequencies_32,
        "Mz": Mz,
        "sigma_1": sigma_1,
        "Sk1": Sk1,
        "Kg": Kg,
        "C_value": C_value,
        "M_value": M_value,
        "Md": Md,
        "sigma": sigma,
        "Sk": Sk,
        "K": K
    }

# 4. 分选性分析
def sorting_analysis(phi_values):
    """计算分选性及分选系数"""
    phi_25, phi_75 = np.percentile(phi_values, [25, 75])
    s0 = phi_75 / phi_25
    sorting_quality = ("Good" if s0 <= 2.5 else "Medium" if s0 <= 4.0 else "Bad")
    s0 = round(s0,1)
    return s0, sorting_quality

# 5. 磨圆度分析
def evaluate_sample_roundness(roundness_values, area_frequencies_12):
    """
    根据磨圆度值和面积频率计算样品的磨圆度。
    :param roundness_values: 每个颗粒的磨圆度值。
    :param area_frequencies_12: 每个粒级的面积频率。
    :return: 样品的磨圆度（中文表示）。
    """
    # 初始化区间面积
    roundness_areas = {interval: 0 for interval in ROUNDNESS_INTERVALS}
    total_area = sum(area_frequencies_12)  # 总面积

    # 遍历所有颗粒的磨圆度值，并将其归入对应的区间
    for roundness, freq in zip(roundness_values, area_frequencies_12):
        for i, interval in enumerate(ROUNDNESS_INTERVALS):
            if interval[0] <= roundness < interval[1]:
                roundness_areas[interval] += freq
                break

    # 计算每个磨圆度区间的面积占比
    roundness_proportions = {interval: (area / total_area) * 100 for interval, area in roundness_areas.items()}

    # 根据规则计算样品的磨圆度
    sorted_areas = sorted(roundness_proportions.items(), key=lambda x: x[1], reverse=True)

    # 面积最大区间是否超过75%
    if sorted_areas[0][1] >= 75:
        primary_interval = sorted_areas[0][0]
        sample_roundness = ROUNDNESS_LABELS[ROUNDNESS_INTERVALS.index(primary_interval)]
    else:
        # 取面积占比最大的两个区间
        primary_interval = sorted_areas[0][0]
        secondary_interval = sorted_areas[1][0]
        primary_label = ROUNDNESS_LABELS[ROUNDNESS_INTERVALS.index(primary_interval)]
        secondary_label = ROUNDNESS_LABELS[ROUNDNESS_INTERVALS.index(secondary_interval)]
        sample_roundness = f"{primary_label}-{secondary_label}"

    return sample_roundness

# 5.1 磨圆度计算
def roundness_analysis(data):
    """计算颗粒的磨圆度"""
    roundness_values = []
    for shape in data.get('shapes', []):
        area = calculate_area(shape['points'])
        perimeter = calculate_perimeter(shape['points'])
        roundness = round((4 * math.pi * area) / (perimeter ** 2), 3) if perimeter > 0 else 0
        roundness_values.append(roundness)
    return roundness_values

# 5.2 不同粒级磨圆度计算
def calculate_roundness_per_grain_size(roundness_values, data, scale):
    """
    计算每个粒径区间的平均磨圆度。
    :param roundness_values: 每个颗粒的磨圆度列表。
    :param data: 样品数据，包括颗粒的点坐标。
    :param scale: 比例尺。
    :return: 每个粒径区间内颗粒磨圆度的平均值。
    """
    roundness_per_interval = {label: [] for label in GRAIN_CLASSES_12}

    for shape, roundness in zip(data.get('shapes', []), roundness_values):
        points = shape['points']
        diameter = calculate_longest_diameter(points, scale)  # 计算最长直径
        phi_value = 0.3815 + 0.9027 * (-math.log2(diameter))

        for i, (min_d, max_d) in enumerate(GRAIN_SIZE_INTERVALS_12[:-1]):  # 去掉“泥”区间
            if min_d <= phi_value < max_d:
                roundness_per_interval[GRAIN_CLASSES_12[i]].append(roundness)
                break

    # 计算每个粒径区间的平均磨圆度
    average_roundness_per_interval = {
        label: (sum(values) / len(values) if len(values) > 0 else 0)
        for label, values in roundness_per_interval.items()
    }

    return average_roundness_per_interval

# 6 颗粒接触方式
def determine_contact_type_with_buffer(p1_polygon, p2_polygon, epsilon=10):
    """
    确定两颗粒之间的接触类型，使用缓冲区考虑容差。
    返回 '无接触', '点接触', '线接触', '复杂接触'。
    """
    # 应用缓冲区
    buffered_p1 = p1_polygon.buffer(epsilon)
    buffered_p2 = p2_polygon.buffer(epsilon)

    # 计算缓冲后的交集
    intersection = buffered_p1.intersection(buffered_p2)

    if intersection.is_empty:
        return 'no'
    elif isinstance(intersection, Polygon):
        # 分析交集多边形的形状
        minx, miny, maxx, maxy = intersection.bounds
        width = maxx - minx
        height = maxy - miny

        # 计算长宽比
        if height == 0 or width == 0:
            aspect_ratio = float('inf')
        else:
            aspect_ratio = width / height

        # 定义一个阈值范围，接近1表示圆形
        lower_threshold = 0.7
        upper_threshold = 1.3

        if lower_threshold <= aspect_ratio <= upper_threshold:
            return 'point'
        else:
            return 'line'
    else:
        # 其他类型的交集（如多点、复合形状）
        return 'no'

def calculate_average_size(particles):
    """
    计算颗粒的平均大小（面积的平方根）。
    """
    sizes = [math.sqrt(p['polygon'].area) for p in particles if p['polygon'].area > 0]
    if not sizes:
        return 1.0  # 默认值
    return sum(sizes) / len(sizes)

def particle_contact_analysis(data, epsilon=10 , visualize=True):
    """
    分析颗粒接触方式并统计主要接触关系，使用缓冲区处理分割误差。
    内部分配唯一ID，不依赖外部提供的ID。

    :param data: 样品数据，包含'形状'等信息。每个形状包含'points'。
    :param epsilon: 缓冲距离，用于处理分割误差。如果为None，则自动计算。
    :param visualize: 是否进行可视化。
    :return: 主要接触类型和接触比例的字典，以及接触对列表。
    """
    shapes = data.get('shapes', [])
    num_shapes = len(shapes)

    if num_shapes < 2:
        return "no contact", {}, []

    # 构建每个颗粒的多边形，并内部分配唯一ID
    particles = []
    polygons = []
    polygon_id_map = {}

    for idx, shape in enumerate(shapes):
        points = shape.get('points', [])
        if len(points) < 3:
            # 无法形成多边形，跳过
            continue
        try:
            polygon = Polygon(points)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)  # 修复无效的多边形
            if polygon.is_empty:
                continue
            particles.append({
                'id': idx,  # 内部分配唯一ID
                'polygon': polygon
            })
            polygons.append(polygon)
            polygon_id_map[id(polygon)] = idx  # 映射id(polygon)到索引
        except Exception as e:
            print(f"Error processing shape at index {idx}: {e}")
            continue

    num_particles = len(particles)

    # 创建空间索引（R-tree）以优化接触判断
    tree = STRtree(polygons)

    contact_types = []
    checked_pairs = set()
    contact_pairs = []  # 存储接触对及其类型

    for i, p1 in enumerate(particles):
        # 查询缓冲后的 p1 可能接触的颗粒
        buffered_p1 = p1['polygon'].buffer(0)
        possible_matches = tree.query(buffered_p1)

        for p2_id in possible_matches:
            if p2_id is None:
                continue  # 未找到对应的ID
            if p2_id <= i:
                continue  # 避免重复处理
            pair = tuple(sorted((p1['id'], p2_id)))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)

            p2 = particles[p2_id]
            contact_type = determine_contact_type_with_buffer(p1['polygon'], p2['polygon'], epsilon=epsilon)
            if contact_type == '复杂':
                contact_type = 'line'  # 简化处理
            contact_types.append(contact_type)
            contact_pairs.append({
                'pair': pair,
                'type': contact_type
            })

    # 计算总可能颗粒对
    all_possible_pairs = num_particles * (num_particles - 1) // 2
    num_contacts = len(contact_types)
    num_no_contacts = all_possible_pairs - num_contacts

    contact_types_with_no_contact = contact_types # + ['无接触'] * num_no_contacts

    # 统计接触类型
    contact_counter = Counter(contact_types_with_no_contact)

    # 计算接触类型比例
    contact_ratios = {k: (v / num_contacts) * 100 for k, v in contact_counter.items()}

    # 按比例排序，找出主要接触类型
    sorted_contacts = sorted(contact_ratios.items(), key=lambda x: x[1], reverse=True)

    primary_contact_type = "no"  # 默认值
    if sorted_contacts:
        primary_type, primary_ratio = sorted_contacts[0]
        if primary_ratio >= 50:
            # 检查是否有其他类型 >=25%
            secondary_types = [k for k, v in contact_ratios.items() if k != primary_type and v >= 25]
            if secondary_types:
                # 根据主要类型和次要类型组合
                # 假设只有一个次要类型满足条件
                secondary_type = secondary_types[0]
                if primary_type == 'point' and secondary_type == 'line':
                    primary_contact_type = 'point-line'
                elif primary_type == 'line' and secondary_type == 'point':
                    primary_contact_type = ('line-point')
                else:
                    # 如果有其他组合，可以自定义
                    primary_contact_type = f"{primary_type}-{secondary_type} contact"
            else:
                # 仅有一种主要接触类型
                primary_contact_type = f"{primary_type} contact"
        else:
            # 没有任何一种类型达到50%，选择比例最高的类型
            primary_contact_type = primary_type

    # 可视化接触（如果启用）
    if visualize:
        fig, ax = plt.subplots(figsize=(10, 10))

        # 绘制所有颗粒
        for p in particles:
            x, y = p['polygon'].exterior.xy
            ax.add_patch(MplPolygon(list(zip(x, y)), fill=True, edgecolor='black', alpha=0.5))

        # 根据接触类型绘制连接线
        for contact in contact_pairs:
            pair = contact['pair']
            contact_type = contact['type']
            p1 = particles[pair[0]]
            p2 = particles[pair[1]]

            # 获取颗粒中心点
            centroid1 = p1['polygon'].centroid
            centroid2 = p2['polygon'].centroid

            # 根据接触类型选择颜色
            if contact_type == 'point':
                color = 'red'
                linewidth = 2
            elif contact_type == 'line':
                color = 'blue'
                linewidth = 2
            else:
                color = 'green'
                linewidth = 2

            # 绘制连接线
            ax.plot([centroid1.x, centroid2.x], [centroid1.y, centroid2.y], color=color, linewidth=linewidth, linestyle='-')

        # 创建图例
        import matplotlib.lines as mlines
        point_contact_line = mlines.Line2D([], [], color='red', label='点接触')
        line_contact_line = mlines.Line2D([], [], color='blue', label='线接触')
        complex_contact_line = mlines.Line2D([], [], color='green', label='无接触')
        ax.legend(handles=[point_contact_line, line_contact_line, complex_contact_line])

        # 标注主要接触类型
        if contact_ratios:
            primary_type = max(contact_ratios, key=contact_ratios.get)
        else:
            primary_type = "no contact"
        plt.title(f"主要接触类型: {primary_type}")
        # plt.xlabel("X")
        # plt.ylabel("Y")
        plt.grid(False)

        # 设置反转 Y 轴
        ax.invert_yaxis()  # 这行代码会反转 Y 轴方向，解决旋转 180 度的问题
        plt.axis('equal')  # 保持比例
        # plt.show()

    return primary_contact_type, contact_ratios, contact_pairs

# 7. 样品命名
def grain_size_naming(area_frequencies_12, relative_proportions):

    gravel_content = sum(area_frequencies_12[:4])  # 砾石内容
    sand_content = sum(area_frequencies_12[4:9])  # 砂内容
    silt_content = sum(area_frequencies_12[9:])  # 粉砂内容

    name = ""

    # Step 1: 初步判断砾石、粉砂含量
    if gravel_content >= 50:
        if sand_content >= 10 and sand_content < 25:
            name = "sand bearing gravel"
        elif sand_content >= 25 and sand_content < 50:
            name = "sandy gravel"
        else:
            name = "gravel"
    elif silt_content >= 50:
        name = "siltstone"

        # Step 2: 如果第一步没有命名，则考虑细粒级别和成分命名
    if name == "":

        component_name = component_naming(relative_proportions)
        # 定义粒度区间
        #grain_classes = ["砾质", "巨砂", "粗砂", "中砂", "细砂", "极细砂", "粉砂质"]
        grain_classes = ["Gravelly", "Giant sand", "Coarse sand", "Medium sand", "Fine sand", "Very fine sand", "Silty"]
        grain_proportions = [
            gravel_content,  # "砾质"
            area_frequencies_12[4],  # "巨砂"
            area_frequencies_12[5],  # "粗砂"
            area_frequencies_12[6],  # "中砂"
            area_frequencies_12[7],  # "细砂"
            area_frequencies_12[8],  # "极细砂"
            silt_content  # "粉砂质"
        ]

        # Step 2.1: 检查是否有三个或三个以上区间 ≥ 25%
        count_25 = sum(1 for proportion in grain_proportions if proportion >= 25)
        if count_25 >= 3:
            name = "inequigranular"

        else:
            # Step 2.2: 如果存在单粒级（五个砂石区间之一）≥ 50%，确定主名
            main_name = ""
            for i in range(1, 6):  # 只检查五个砂石区间（"巨砂" ~ "极细砂"）
                if grain_proportions[i] >= 50:
                    main_name = grain_classes[i]
                    break  # 找到主名后退出循环

            if main_name:
                name = main_name

                # Step 2.2.1: 添加副名（如果存在其他区间 ≥ 25%）
                sub_names = []
                for i, proportion in enumerate(grain_proportions):
                    if i != grain_classes.index(main_name) and proportion >= 25:  # 排除主名对应的区间
                        sub_names.append(grain_classes[i])

                if sub_names:
                    name = " ".join(sub_names) + ' ' + name

            else:
                # Step 2.3: 如果不存在单粒级 ≥ 50%，判断是否存在两个 ≥ 25% 的区间
                significant_classes = [
                    grain_classes[i] for i, proportion in enumerate(grain_proportions) if proportion >= 25
                ]

                if len(significant_classes) == 2:
                    # 判断是否相邻
                    indices = [grain_classes.index(c) for c in significant_classes]
                    if max(indices) - min(indices) == 1:
                        # 相邻，用 "-" 命名
                        name = "-".join(significant_classes)
                    else:
                        # 不相邻，命名为 "inequigranular"
                        name = "inequigranular"
                else:
                    # 如果无法满足其他条件
                    name = "inequigranular"

    # 对最终的 name 进行后处理
    name = finalize_name(name)
    name = name + ' ' + component_name

    return name


def component_naming(relative_proportions):
    """
    根据组分含量规则命名样品。
    :param relative_proportions: 相对含量字典，包含石英、长石和岩屑的百分比。
    :return: 样品的成分命名。
    """
    # 提取石英、长石和岩屑的相对含量
    # quartz_content = relative_proportions.get("石英", 0)
    # feldspar_content = relative_proportions.get("长石", 0)
    # lithic_content = relative_proportions.get("岩屑", 0)

    quartz_content = relative_proportions.get("quartz", 0)
    feldspar_content = relative_proportions.get("feldspar", 0)
    lithic_content = relative_proportions.get("", 0)

    # # 规则①：石英含量≥90%，命名为“石英砂岩”
    # if quartz_content >= 90:
    #     return "石英砂岩"
    #
    # # 规则②：石英含量≥75%，计算长石/岩屑的相对比例
    # if quartz_content >= 75:
    #     if feldspar_content / lithic_content >= 1:
    #         return "长石石英砂岩"
    #     else:
    #         return "岩屑石英砂岩"
    #
    # # 规则③：石英含量＜75%，计算长石/岩屑的相对比例
    # if feldspar_content / lithic_content >= 3:
    #     return "长石砂岩"
    # elif 1 <= feldspar_content / lithic_content < 3:
    #     return "岩屑长石砂岩"
    # elif 1 / 3 <= feldspar_content / lithic_content < 1:
    #     return "长石岩屑砂岩"
    # else:
    #     return "岩屑砂岩"

    # Rule ①: If quartz content ≥ 90%, name it as "quartz sandstone"
    if quartz_content >= 90:
        return "quartz sandstone"

    # Rule ②: If quartz content ≥ 75%, calculate the relative ratio of feldspar/lithic
    if quartz_content >= 75:
        if lithic_content==0 or feldspar_content / lithic_content >= 1:
            return "feldspar quartz sandstone"
        else:
            return "lithic quartz sandstone"

    # Rule ③: If quartz content < 75%, calculate the relative ratio of feldspar/lithic
    if lithic_content==0 or feldspar_content / lithic_content >= 3:
        return "feldspar sandstone"
    elif 1 <= feldspar_content / lithic_content < 3:
        return "lithic feldspar sandstone"
    elif 1 / 3 <= feldspar_content / lithic_content < 1:
        return "feldspar lithic sandstone"
    else:
        return "lithic sandstone"


def terrigenous_clast_naming(relative_proportions):
    # 计算端元含量
    quartz_content = relative_proportions['石英']
    feldspar_content = relative_proportions.get("斜长石", 0) + relative_proportions.get("钾长石", 0)
    lithic_content = relative_proportions.get("岩屑-沉积岩", 0)
    total_content = quartz_content + feldspar_content + lithic_content

    # 计算相对百分比
    quartz_percent = (quartz_content / total_content) * 100 if total_content > 0 else 0
    feldspar_percent = (feldspar_content / total_content) * 100 if total_content > 0 else 0
    lithic_percent = (lithic_content / total_content) * 100 if total_content > 0 else 0

    # 命名规则
    # if quartz_percent >= 90:
    #     clast_name = "石英砂岩"
    # elif quartz_percent >= 75:
    #     clast_name = "长石石英砂岩" if feldspar_percent >= lithic_percent else "岩屑石英砂岩"
    # elif feldspar_percent >= 75:
    #     clast_name = "长石砂岩"
    # elif lithic_percent >= 75:
    #     clast_name = "岩屑砂岩"
    # elif feldspar_percent / lithic_percent >= 3:
    #     clast_name = "长石砂岩"
    # elif 1 <= feldspar_percent / lithic_percent < 3:
    #     clast_name = "岩屑长石砂岩"
    # elif 1 / 3 <= feldspar_percent / lithic_percent < 1:
    #     clast_name = "长石岩屑砂岩"
    # else:
    #     clast_name = "岩屑砂岩"
    #
    # return clast_name

    if quartz_percent >= 90:
        clast_name = "quartz sandstone"
    elif quartz_percent >= 75:
        clast_name = "feldspar quartz sandstone" if feldspar_percent >= lithic_percent else "lithic quartz sandstone"
    elif feldspar_percent >= 75:
        clast_name = "feldspar sandstone"
    elif lithic_percent >= 75:
        clast_name = "lithic sandstone"
    elif lithic_content==0 or feldspar_percent / lithic_percent >= 3:
        clast_name = "feldspar sandstone"
    elif 1 <= feldspar_percent / lithic_percent < 3:
        clast_name = "lithic feldspar sandstone"
    elif 1 / 3 <= feldspar_percent / lithic_percent < 1:
        clast_name = "feldspar lithic sandstone"
    else:
        clast_name = "lithic sandstone"

    return clast_name


# 最终命名后处理
def finalize_name(name):
    # 将所有的“砂”替换为“粒”
    name = name.replace("砂", "粒")
    # 如果有两个“粒”，去掉第一个“粒”
    if name.count("粒") >= 2:
        first_index = name.find("粒")
        name = name[:first_index] + name[first_index + 1:]
    return name


# 绘图函数
def plot_roundness_per_grain_size(average_roundness_per_interval, output_dir):
    """
    绘制每个粒径区间的平均磨圆度分布图。
    :param average_roundness_per_interval: 每个粒径区间的平均磨圆度。
    :param output_dir: 输出文件夹路径。
    """
    grain_classes = list(average_roundness_per_interval.keys())
    average_roundness_values = list(average_roundness_per_interval.values())

    plt.figure(figsize=(10, 6))
    plt.bar(grain_classes, average_roundness_values, color='skyblue')
    plt.xlabel("Grain Size Interval")
    plt.ylabel("Average Roundness")
    plt.title("Roundness Distribution of Different Grain Sizes")
    plt.ylim(0, 1.0)  # 确保纵轴范围在0到1之间
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, "roundness_distribution.png"), bbox_inches='tight', dpi=300)
    plt.close()

def plot_grain_size_combined_and_pie(area_frequencies_12, cumulative_frequencies_12, output_dir):
    """
    绘制粒径-面积含量占比分布图（折线图）和不同粒级颗粒含量图（柱状图）合并为一张图，
    另单独绘制一个饼图。

    :param area_frequencies_12: 12个粒径区间的面积频率表。
    :param cumulative_frequencies_12: 12个粒径区间的累积频率表。
    :param output_dir: 输出文件夹。
    """
    # 定义中文粒径区间
    # 确保数据长度一致
    if len(area_frequencies_12) != len(GRAIN_CLASSES_12):
        raise ValueError("粒径区间频率数据长度与粒径区间标签长度不一致")

    # 合并图：粒径-面积含量占比分布图（折线图）和不同粒级颗粒含量图（柱状图）
    plt.figure(figsize=(14, 8))

    # 1. 折线图：粒径-面积含量占比分布图
    plt.plot(GRAIN_CLASSES_12, cumulative_frequencies_12, marker='o', linestyle='-', label="Cumulative frequency(%)", color="blue")

    # 2. 柱状图：不同粒级颗粒含量图
    plt.bar(GRAIN_CLASSES_12, area_frequencies_12, alpha=0.6, label="Area Frequency(%)", color="orange")

    # 图表设置
    plt.xlabel("Grain size intervals")
    plt.ylabel("Frequency (%)")
    plt.title("Particle size-area content proportion and particle content plots of different particle sizes")
    plt.axhline(50, color="gray", linestyle="--", alpha=0.7, label="Median(50%)")
    plt.legend()

    # 保存合并图
    combined_output_path = os.path.join(output_dir, "Particle size area content and particle content of different particle sizes_Combined figures.png")
    plt.savefig(combined_output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"粒径面积含量和不同粒级颗粒含量合并图已保存到: {combined_output_path}")

    # 饼图：不同粒级颗粒含量
    # 过滤掉频率为 0 的区间
    filtered_frequencies = [freq for freq in area_frequencies_12 if freq > 0]
    filtered_classes = [cls for freq, cls in zip(area_frequencies_12, GRAIN_CLASSES_12) if freq > 0]

    if filtered_frequencies:
        plt.figure(figsize=(8, 8))
        plt.pie(filtered_frequencies, labels=filtered_classes, autopct='%1.1f%%', startangle=140)
        plt.title("Particle Content Diagram at Different Grain Sizes")

        # 保存饼图
        pie_output_path = os.path.join(output_dir, "Particle Content Diagram at Different Grain Levels.png")
        plt.savefig(pie_output_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"不同粒级颗粒含量饼图已保存到: {pie_output_path}")

def plot_grain_size_distribution(cumulative_frequencies_12, output_dir):
    """
    绘制粒径-面积含量占比分布图。
    横坐标为粒径范围（中文），纵坐标为累计频率。
    """
    #grain_classes_12 =

    plt.figure(figsize=(10, 6))
    plt.plot(GRAIN_CLASSES_12, cumulative_frequencies_12, marker='o', linestyle='-', label="面积含量 (%)")
    plt.xlabel("Particle size range")
    plt.ylabel("Area content (%)")
    plt.title("grain size cumulative distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grain_size_cumulative_distribution.png"))
    plt.close()
    print(f"粒径-面积含量占比分布图已保存到 {output_dir}")


def plot_grain_size_frequency_distribution(area_frequencies_12, output_dir):
    """
    绘制不同粒级颗粒含量图（柱状图）。
    横坐标为粒径范围（中文），纵坐标为面积频率。
    """
    plt.figure(figsize=(10, 6))
    plt.bar(GRAIN_CLASSES_12, area_frequencies_12, color='skyblue', alpha=0.7, label="area frequency")
    plt.xlabel("Grain size interval")
    # 设置 x 轴刻度标签旋转 45 度
    plt.xticks(rotation=45)
    plt.ylabel("area frequency(%)")
    plt.title("Grain Size Frequency Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grain_size_frequency_distribution.png"))
    plt.close()
    print(f"不同粒级颗粒含量图已保存到 {output_dir}")

def weighted_average(values, weights):
    """Calculate weighted average of values with corresponding weights."""
    weighted_sum = sum(value * weight for value, weight in zip(values, weights))
    total_weight = sum(weights)
    return weighted_sum / total_weight if total_weight != 0 else 0

def plot_grain_area_and_roundness_distribution(
    area_frequencies_12, cumulative_frequencies_12, average_roundness_per_size, output_dir
):
    """
    绘制陆源碎屑不同粒级面积及磨圆度分布图（部分区间合并）。
    :param area_frequencies_12: 12区间的面积频率。
    :param cumulative_frequencies_12: 12区间的累计面积频率。
    :param average_roundness_per_size: 每个粒径区间的平均磨圆度（字典）。
    :param output_dir: 输出目录。
    """
    # 确保 average_roundness_per_size 是一个字典
    if not isinstance(average_roundness_per_size, dict):
        print("Error: average_roundness_per_size 必须是一个字典。")
        return

    # 合并区间和保留区间
    grain_classes_merged = [
        'Gravel',  # 砾石 (前四个区间合并)
        GRAIN_CLASSES_12[4],  # 第五个区间
        'Coarse Sand',  # 粗砂 (第五和第六个区间合并)
        GRAIN_CLASSES_12[6],  # 第七个区间
        GRAIN_CLASSES_12[7],  # 第八个区间
        'Silt'  # 粉砂 (倒数第二和第三个区间合并)
    ]
    merged_area_frequencies = [
        sum(area_frequencies_12[:4]),  # 砾石
        area_frequencies_12[4],  # 第五个区间
        sum(area_frequencies_12[5:7]),  # 粗砂
        area_frequencies_12[7],  # 第七个区间
        area_frequencies_12[8],  # 第八个区间
        sum(area_frequencies_12[-3:-1])  # 粉砂
    ]
    merged_cumulative_frequencies = [
        max(cumulative_frequencies_12[:4]),  # 砾石
        cumulative_frequencies_12[4],  # 第五个区间
        max(cumulative_frequencies_12[5:7]),  # 粗砂
        cumulative_frequencies_12[7],  # 第七个区间
        cumulative_frequencies_12[8],  # 第八个区间
        max(cumulative_frequencies_12[-3:-1])  # 粉砂
    ]

    merged_average_roundness = [
        weighted_average([average_roundness_per_size.get(size, 0) for size in GRAIN_CLASSES_12[:4]],
                         area_frequencies_12[:4]),
        average_roundness_per_size.get(GRAIN_CLASSES_12[4], 0),
        weighted_average([average_roundness_per_size.get(size, 0) for size in GRAIN_CLASSES_12[5:7]],
                         area_frequencies_12[5:7]),
        average_roundness_per_size.get(GRAIN_CLASSES_12[7], 0),
        average_roundness_per_size.get(GRAIN_CLASSES_12[8], 0),
        weighted_average([average_roundness_per_size.get(size, 0) for size in GRAIN_CLASSES_12[-3:-1]],
                         area_frequencies_12[-3:-1])
    ]

    # 创建画布
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # 左侧纵轴：累计面积频率的折线图
    ax1.plot(
        grain_classes_merged,
        merged_cumulative_frequencies,
        color="blue",
        marker="o",
        linestyle="-",
        linewidth=2,
        label="Cumulative Frequency(%)"
    )
    ax1.set_ylabel("Cumulative frequency(%)", fontsize=12, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_ylim(0, 100)

    # 右侧纵轴：平均磨圆度的柱状图
    ax2 = ax1.twinx()
    ax2.bar(
        grain_classes_merged,
        merged_average_roundness,
        color="orange",
        alpha=0.7,
        label="Average roundness",
        width=0.6
    )
    ax2.set_ylabel("Average roundness(0~1)", fontsize=12, color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    ax2.set_ylim(0, 1)

    # 添加标题和图例
    ax1.set_xlabel("Grain size interval", fontsize=14)
    ax1.set_title("Area and Roundness Distribution After Merging Intervals", fontsize=16)
    ax1.tick_params(axis='x', rotation=45)

    fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + bar_labels, loc="upper left")

    # 保存图像
    output_path = os.path.join(output_dir, "Merged Area and Roundness Distribution.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"合并后的陆源碎屑不同粒级面积及磨圆度分布图已保存到: {output_path}")


#粒度分析综合图
def plot_comprehensive_grain_size_chart(phi_ranges_32, area_frequencies_32, cumulative_frequencies_32, phi_values, output_dir):
    """
    绘制粒度分析综合图：包括频率图、累积频率图和正态概率累积图。
    """
    # 计算32区间的中点，用于绘制横轴
    phi_midpoints = [(start + end) / 2 for start, end in phi_ranges_32]
    phi_midpoints[0] = -9
    phi_midpoints[-1] = 9
    # 正态概率累积图：计算累积概率
    sorted_phi_values = sorted(phi_values)
    n = len(sorted_phi_values)
    probabilities = [(i + 1 - 0.5) / n * 100 for i in range(n)]

    # 创建画布
    plt.figure(figsize=(20, 8))

    # 第1部分：频率图（柱状图）
    plt.bar(phi_midpoints, list(area_frequencies_32.values()),
            width=0.1, alpha=0.7, label="Frequency (Area percentage)", color="skyblue")

    # 第2部分：累积频率图（折线图）
    plt.plot(phi_midpoints, cumulative_frequencies_32,
             marker='o', linestyle='-',
             label="Cumulative Frequency (Cumulative Area pencentage)", color="orange")

    # 第3部分：正态概率累积图（散点图）
    plt.scatter(sorted_phi_values, probabilities, label="Normal Probability Accumulation Graph (Probability Percentage)", color="green")

    # 添加辅助线
    plt.axhline(50, color="gray", linestyle="--", alpha=0.7, label="Median Line (50%)")

    # 设置横坐标范围和刻度
    plt.xlim(-10, 10)  # 固定横坐标范围为 -10 到 10
    plt.xticks(range(-10, 11, 2))  # 设置刻度为 -10 到 10，步长为 2

    # 图例、标题和轴标签
    plt.xlabel("φ value")
    plt.ylabel("Percentage (%)")
    plt.title("Grain size analysis summary")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # 保存图表
    output_path = os.path.join(output_dir, "grain_distribution_summary.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"粒度分析综合图已生成: {output_path}")


def generate_jsons(
        content_analysis_results,
        grain_results,
        results,
        average_roundness_per_size,
        output_dir
):
    """
    整合生成所有所需的 JSON 文件。
    """

    # Helper function: 将 NumPy 数组转换为 Python 列表
    def convert_to_list(data):
        if isinstance(data, np.ndarray):
            return data.tolist()  # 转换为列表
        if isinstance(data, dict):
            return {key: convert_to_list(value) for key, value in data.items()}  # 递归处理字典
        if isinstance(data, list):
            return [convert_to_list(item) for item in data]  # 递归处理列表
        return data  # 其他类型不变


    json_summary = {}

    # 1. 一级分类成分分布图 JSON
    first_level_composition = {
        "absolute_proportion": content_analysis_results["absolute_proportions_first"],
        "relative_proportion": content_analysis_results["relative_proportions_first"]
    }
    first_level_filename = "first_level_composition.json"
    with open(os.path.join(output_dir, "first_level_composition.json"), "w", encoding="utf-8") as f:
        json.dump(first_level_composition, f, ensure_ascii=False, indent=4)
    json_summary["first_level_composition"] = first_level_composition

    # 2. 二级分类成分分布图 JSON
    second_level_composition = {
        "absolute_proportion": content_analysis_results["absolute_proportions_second"],
        "relative_proportion": content_analysis_results["relative_proportions_second"]
    }
    second_level_filename = "second_level_composition.json"
    with open(os.path.join(output_dir, "second_level_composition.json"), "w", encoding="utf-8") as f:
        json.dump(second_level_composition, f, ensure_ascii=False, indent=4)
    json_summary["second_level_composition"] = second_level_composition

    # 3. 粒径-面积含量累积分布图 JSON
    area_frequencies = {}
    cumulative_frequencies = {}
    area_frequencies_list = convert_to_list(grain_results["area_frequencies_12"])
    cumulative_frequencies_list = convert_to_list(grain_results["cumulative_frequencies_12"])
    for i, grain_class in enumerate(GRAIN_CLASSES_12):
        area_frequencies[grain_class] = area_frequencies_list[i]
        cumulative_frequencies[grain_class] = cumulative_frequencies_list[i]
        
    grain_size_cumulative = {
        "area_frequencies": area_frequencies,
        "cumulative_frequencies": cumulative_frequencies
    }
    with open(os.path.join(output_dir, "grain_size_distribution.json"), "w", encoding="utf-8") as f:
        json.dump(grain_size_cumulative, f, ensure_ascii=False, indent=4)
    json_summary["grain_size_cumulative_distribution"] = grain_size_cumulative

    # 4. 粒级分布图 JSON
    grain_level_distribution_filename = "grain_level_distribution.json"
    grain_level_distribution = {
        "area_frequencies": area_frequencies
        
    }
    with open(os.path.join(output_dir, "grain_level_distribution.json"), "w", encoding="utf-8") as f:
        json.dump(grain_level_distribution, f, ensure_ascii=False, indent=4)
    json_summary["grain_level_distribution"] = grain_level_distribution

    # 5. 粒度分析 JSON
    grain_analysis_filename = "grain_analysis.json"
    grain_analysis = {
        "grain_analysis": {
            "graphical_parameters": {
                "Mz (Mean Size)": grain_results["Mz"],  # 平均粒径
                "σ1 (Sorting Coefficient)": grain_results["sigma_1"],  # 分选系数
                "Sk1 (Skewness)": grain_results["Sk1"],  # 偏态系数
                "Kg (Kurtosis)": grain_results["Kg"]  # 峰态系数
            },
            "moment_parameters": {
                "Md (Median Diameter)": grain_results["Md"],  # 中值粒径
                "σ (Standard Deviation)": grain_results["sigma"],  # 标准差
                "Sk (Skewness)": grain_results["Sk"],  # 偏态
                "K (Kurtosis)": grain_results["K"]  # 峰态
            }
        }
    }
    with open(os.path.join(output_dir, "grain_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(grain_analysis, f, ensure_ascii=False, indent=4)
    json_summary["grain_analysis"] = grain_analysis

    # 6. 综合图 JSON（频率图、累积频率图、正态概率累积图）
    comprehensive_filename = "comprehensive_distribution.json"
    phi_midpoints = [(start + end) / 2 for start, end in GRAIN_SIZE_INTERVALS_32]
    sorted_phi_values = sorted(grain_results['phi_values'])
    phi_midpoints[0] = -9
    phi_midpoints[-1] = 9
    n = len(sorted_phi_values)
    probabilities = [(i + 1 - 0.5) / n * 100 for i in range(n)]
    comprehensive_data = {
        "frequencies_graph": {
            "φ_interval": [f"({start}, {end})" for start, end in GRAIN_SIZE_INTERVALS_32],
            "midpoints": phi_midpoints,
            "area_frequencies": list(grain_results["area_frequencies_32"].values())
        },
        "cumulative_frequencies_graph": {
            "φ_interval": [f"({start}, {end})" for start, end in GRAIN_SIZE_INTERVALS_32],
            "midpoints": phi_midpoints,
            "cumulative_frequencies": list(grain_results["cumulative_frequencies_32"])
        },
        "normal_probability_chart": {
            "φ_value": sorted_phi_values,
            "probabilities": probabilities
        }
    }
    with open(os.path.join(output_dir, "comprehensive_chart.json"), "w", encoding="utf-8") as f:
        json.dump(convert_to_list(comprehensive_data), f, ensure_ascii=False, indent=4)
    json_summary["comprehensive_distribution"] = comprehensive_data

    # 7. 计算结果 JSON
    results_filename = "calculation_results.json"
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(convert_to_list(results), f, ensure_ascii=False, indent=4)
    json_summary["calculation_results"] = results

    # 8. 不同粒级磨圆度分布图 JSON
    roundness_distribution_filename = "roundness_distribution.json"
    roundness_distribution = {
        "roundness_distribution": convert_to_list(average_roundness_per_size)
    }
    with open(os.path.join(output_dir, "roundness_distribution.json"), "w", encoding="utf-8") as f:
        json.dump(roundness_distribution, f, ensure_ascii=False, indent=4)
    json_summary["roundness_distribution"] = roundness_distribution

    # 9. 不同粒级面积及磨圆度分布图 JSON
    area_and_roundness_distribution_filename = "area_and_roundness_distribution.json"
    area_and_roundness_distribution = {
        "area_and_roundness_distribution": {
            "average_roundness": convert_to_list(average_roundness_per_size),
            "cumulative_area_frequencies": convert_to_list(grain_results["cumulative_frequencies_12"])
        }
    }
    with open(os.path.join(output_dir, "area_and_roundness_distribution.json"), "w", encoding="utf-8") as f:
        json.dump(area_and_roundness_distribution, f, ensure_ascii=False, indent=4)
    json_summary["area_and_roundness_distribution"] = area_and_roundness_distribution


    # 10. 保存汇总 JSON
    # summary_filename = "summary.json"
    # with open(os.path.join(output_dir, summary_filename), "w", encoding="utf-8") as f:
    #     json.dump(json_summary, f, ensure_ascii=False, indent=4)

    print(f"所有 JSON 文件已生成并保存到: {output_dir}")
    return json_summary  # 返回汇总字典

def output_results(
    max_diameter,
    main_grain_range,
    interval_proportions,
    absolute_proportions_fine,
    relative_proportions_fine,
    absolute_proportions_coarse,
    relative_proportions_coarse,
    s0,
    sorting_quality,
    sample_roundness,  # 新增参数
    average_roundness_per_size,
    primary_contact_type,
    contact_ratios,
    final_name,
    output_dir
):
    """
    输出所有结果，包括打印和保存为CSV或JSON文件。
    """
    results = {
        "最大粒径": max_diameter,
        "主要粒径区间": main_grain_range,
        "粒径区间的面积占比": interval_proportions,
        "绝对含量（第二级）": absolute_proportions_fine,
        "相对含量（第二级）": relative_proportions_fine,
        "绝对含量（第一级）": absolute_proportions_coarse,
        "相对含量（第一级）": relative_proportions_coarse,
        "分选性": sorting_quality,
        "分选系数 S0": s0,
        "磨圆度分布": sample_roundness,
        "平均磨圆度（不同粒级）": average_roundness_per_size,
        "主要接触方式": primary_contact_type,
        "接触比例": contact_ratios,
        "样品命名": final_name
    }

    # 打印结果
    print("\n==== 计算结果 ====\n")
    for key, value in results.items():
        print(f"{key}:")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        elif isinstance(value, list):
            for item in value:
                print(f"  {item}")
        else:
            print(f"  {value}")
        print("\n")


    # 保存结果为CSV文件
    csv_path = os.path.join(output_dir, "综合结果.csv")
    with open(csv_path, mode="w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in results.items():
            writer.writerow([key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])

    print(f"综合结果已保存到: {csv_path}")


def create_output_folder(file_path):
    """
    创建一个输出文件夹，用于存储所有输出结果。
    """
    folder_name = os.path.splitext(os.path.basename(file_path))[0] + "_results"
    output_dir = os.path.join(os.path.dirname(file_path), folder_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# 主函数
def generate_report(scale, estimated_basal_content,json_file_path):

    try:
        # 创建输出目录
        output_dir = create_output_folder(json_file_path)
        data = load_data(json_file_path)
        total_image_area = data['imageWidth'] * data['imageHeight']

        # 1. 粒径分析
        max_diameter, main_grain_range, interval_proportions = calculate_main_grain_range(data, scale)

        # 2. 组分分析
        # 调用函数，获取一级分类和二级分类的组分分析结果
        content_analysis_results = component_content_analysis(data, scale, output_dir)

        # 查看返回的结果（一级和二级分类的绝对/相对含量）
        # print("一级分类绝对含量:", content_analysis_results["absolute_proportions_first"])
        # print("一级分类相对含量:", content_analysis_results["relative_proportions_first"])
        # print("二级分类绝对含量:", content_analysis_results["absolute_proportions_second"])
        # print("二级分类相对含量:", content_analysis_results["relative_proportions_second"])

        # 3. 粒度分析
        grain_results = grain_size_analysis(data, scale, estimated_basal_content, output_dir)
        phi_values = grain_results['phi_values']  # 提取phi_values
        area_frequencies_12 = grain_results['area_frequencies_12']
        area_frequencies_32 = grain_results['area_frequencies_32']

        # 4.分选性分析
        s0, sorting_quality = sorting_analysis(phi_values)
        plot_grain_size_combined_and_pie(
            area_frequencies_12=grain_results['area_frequencies_12'],
            cumulative_frequencies_12=grain_results['cumulative_frequencies_12'],
            output_dir=output_dir
        )
        #粒径-面积含量占比分布图
        plot_grain_size_distribution(
            cumulative_frequencies_12=grain_results['cumulative_frequencies_12'],
            output_dir=output_dir
        )
        #不同粒级颗粒含量图
        plot_grain_size_frequency_distribution(
            area_frequencies_12=grain_results['area_frequencies_12'],
            output_dir=output_dir
        )

        # 5.磨圆度分析
        roundness_values = roundness_analysis(data)  # 调用磨圆度分析函数
        # 计算每个粒径区间的平均磨圆度（不包括泥）
        average_roundness_per_interval = calculate_roundness_per_grain_size(roundness_values, data, scale)
        # 绘制每个粒径区间的平均磨圆度分布图
        plot_roundness_per_grain_size(average_roundness_per_interval, output_dir)
        # 计算样品的磨圆度
        sample_roundness = evaluate_sample_roundness(roundness_values, area_frequencies_12)

        # 6.接触方式分析
        primary_contact_type, contact_ratios, pairs = particle_contact_analysis(data)

        # 7.样品命名
        final_name = grain_size_naming(area_frequencies_12, content_analysis_results["relative_proportions_first"])

        plot_roundness_per_grain_size(average_roundness_per_interval,output_dir)

        # 绘制磨圆度和面积综合图
        plot_grain_area_and_roundness_distribution(
            grain_results['area_frequencies_12'],
            grain_results['cumulative_frequencies_12'],
            average_roundness_per_interval,
            output_dir=output_dir
        )

        # 绘制粒度分析综合图
        plot_comprehensive_grain_size_chart(
            phi_ranges_32=GRAIN_SIZE_INTERVALS_32,
            area_frequencies_32=grain_results['area_frequencies_32'],
            cumulative_frequencies_32=grain_results['cumulative_frequencies_32'],
            phi_values=grain_results['phi_values'],
            output_dir=output_dir
        )

        #生成json
        summary_content = generate_jsons(
            content_analysis_results=content_analysis_results,
            grain_results=grain_results,
            results={
                "max_diameter": max_diameter,
                "main_grain_size_interval": main_grain_range,
                "area_proportion_of_grain_size_interval": interval_proportions,
                "absolute_content_second_level": content_analysis_results["absolute_proportions_second"],
                "relative_content_second_level": content_analysis_results["relative_proportions_second"],
                "absolute_content_first_level": content_analysis_results["absolute_proportions_first"],
                "relative_content_first_level": content_analysis_results["relative_proportions_first"],
                "sorting_quality": sorting_quality,
                "sorting_coefficient_s0": s0,
                "roundness_distribution": roundness_values,
                "sample_roundness": sample_roundness,
                "average_roundness_per_grain_sizes": average_roundness_per_interval,
                "primary_contact_mode": primary_contact_type,
                "contact_ratio": contact_ratios,
                "sample_naming": final_name
            },
            average_roundness_per_size=average_roundness_per_interval,
            output_dir=output_dir
        )

        output_results(
            max_diameter=max_diameter,
            main_grain_range=main_grain_range,
            interval_proportions=interval_proportions,
            absolute_proportions_fine=content_analysis_results["absolute_proportions_second"],
            relative_proportions_fine=content_analysis_results["relative_proportions_second"],
            absolute_proportions_coarse=content_analysis_results["absolute_proportions_first"],
            relative_proportions_coarse=content_analysis_results["relative_proportions_first"],
            s0=s0,
            sorting_quality=sorting_quality,
            sample_roundness=sample_roundness,
            average_roundness_per_size=average_roundness_per_interval,
            primary_contact_type=primary_contact_type,
            contact_ratios=contact_ratios,
            final_name=final_name,
            output_dir=output_dir
        )

        return {'status':True,'res':summary_content}
    except Exception as e:
        traceback.print_exc()
        error_str = traceback.format_exc()
        return {'status':False,'res':error_str}


if "__main__" == __name__:
    # 默认值
    scale = 535 # 263 / 0.5
    # a388:263:0.5mm
    # a356：468：0.2mm
    estimated_basal_content = 5.0  # 默认目估杂基含量
    json_file_path = "./uploadFiles/a388-.json"  # 替换为你的默认路径

    final_dict = generate_report(scale, estimated_basal_content, json_file_path)

    # dict1={'key':np.float64(1.0)}
    # json.dumps(final_dict)
                                 