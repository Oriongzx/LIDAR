# coding: utf-8
import laspy
import CSF
import numpy as np
import open3d as o3d
# 440882004002000201800 - Cloud.las
# 1. 读取 LAS 文件
inFile = laspy.read(r"D:\激光雷达\2点云滤波\440882004002000201800 - Cloud.las")  # 读取 .las 文件
points = inFile.points
xyz = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()  # 提取 x, y, z 坐标

# 2. 使用 CSF 进行点云滤波
csf = CSF.CSF()

# 设置参数
csf.params.bSloopSmooth = False
csf.params.cloth_resolution = 0.5
csf.params.rigidness = 3
# 以下参数一般不用调整
csf.params.time_step = 0.65
csf.params.class_threshold = 0.5
csf.params.interations = 500

csf.setPointCloud(xyz)

# 创建两个列表分别存储地面点和非地面点的索引
ground = CSF.VecInt()  # 存储地面点索引
non_ground = CSF.VecInt()  # 存储非地面点索引

# 执行滤波
csf.do_filtering(ground, non_ground)

# 3. 提取地面点和非地面点
ground_points = xyz[np.array(ground)]  # 提取地面点
non_ground_points = xyz[np.array(non_ground)]  # 提取非地面点

# 4. 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)  # 设置原始点云

ground_pcd = o3d.geometry.PointCloud()
ground_pcd.points = o3d.utility.Vector3dVector(ground_points)  # 设置地面点云

non_ground_pcd = o3d.geometry.PointCloud()
non_ground_pcd.points = o3d.utility.Vector3dVector(non_ground_points)  # 设置非地面点云

# 5. 显示点云
# 显示原始点云
o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

# 显示地面点云
o3d.visualization.draw_geometries([ground_pcd], window_name="Ground Points")

# 显示非地面点云
o3d.visualization.draw_geometries([non_ground_pcd], window_name="Non-Ground Points")

# 6. 保存滤波后的地面点和非地面点到新的 LAS 文件
outFile = laspy.LasData(inFile.header)
outFile.points = points[np.array(ground)]  # 提取地面点并保存
outFile.write(r"ground.las")
outFile.points = points[np.array(non_ground)]  # 提取地面点并保存
outFile.write(r"non_ground.las")