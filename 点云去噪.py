import open3d as o3d
import numpy as np
import platform
import sys
import time

# 打印运行环境信息
def print_environment_info():
    print("=== 环境信息 ===")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python 版本: {platform.python_version()}")
    print(f"Python 可执行文件: {sys.executable}")
    print(f"处理器架构: {platform.architecture()[0]}")
    print(f"处理器: {platform.processor()}")
    print(f"系统平台: {platform.platform()}")

# 读取点云数据
def load_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

# 计算每个点到其邻居的平均距离
def compute_average_distances(pcd, mean_k=50):
    points = np.asarray(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    avg_distances = []

    for point in points:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, mean_k)
        neighbors = points[idx[1:], :]
        distances = np.linalg.norm(neighbors - point, axis=1)
        avg_distance = np.mean(distances)
        avg_distances.append(avg_distance)

    return np.array(avg_distances)

# 基于统计去噪
def denoise_point_cloud(pcd, mean_k=50, std_dev_multiplier=1.0):
    avg_distances = compute_average_distances(pcd, mean_k)

    mean_distance = np.mean(avg_distances)
    std_dev_distance = np.std(avg_distances)

    filtered_points = []
    for i, distance in enumerate(avg_distances):
        if abs(distance - mean_distance) <= std_dev_multiplier * std_dev_distance:
            filtered_points.append(np.asarray(pcd.points)[i])

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    return filtered_pcd

# 显示点云
def display_point_cloud(pcd, title="Point Cloud"):
    o3d.visualization.draw_geometries([pcd], window_name=title)

# 主函数
def main():
    start_time = time.time()  # 记录开始时间

    # 打印运行环境信息
    print_environment_info()

    # 文件路径（请修改为你自己的文件路径）
    input_pcd_path = r"table_scene_lms400.pcd"

    # 读取点云
    cloud = load_pcd(input_pcd_path)

    # 打印去噪前的点云数量
    print(f"原始点云有 {len(cloud.points)} 个点")

    # 显示去噪前的点云
    display_point_cloud(cloud, "Original Point Cloud")

    # 进行去噪处理
    cloud_filtered = denoise_point_cloud(cloud)

    # 打印去噪后的点云数量
    print(f"去噪后点云有 {len(cloud_filtered.points)} 个点")

    # 显示去噪后的点云
    display_point_cloud(cloud_filtered, "Filtered Point Cloud")

    # 保存去噪后的点云到文件
    o3d.io.write_point_cloud("table_scene_lms400_filtered.pcd", cloud_filtered)

    end_time = time.time()  # 记录结束时间
    print(f"\n运行时间: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
