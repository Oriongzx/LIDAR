import torch
import numpy as np
from pointnet2_cls_ssg import get_model  # 替换为你的模型文件


# 1. 加载训练好的模型
def load_model(model_path, num_class):
    model = get_model(num_class=num_class, normal_channel=False)
    checkpoint = torch.load(model_path)  # 加载权重文件
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
    model.eval()  # 设置为评估模式
    return model


# 2. 准备点云数据
def load_point_cloud(file_path):
    # 读取点云数据（此处假设数据是以 (x, y, z, nx, ny, nz) 的形式存储）
    point_cloud = np.loadtxt(file_path, delimiter=',')  # 修改为你实际的分隔符

    points = point_cloud[:, :3]  # 点坐标 (x, y, z)
    normals = point_cloud[:, 3:]  # 点法线 (nx, ny, nz)

    data = points  # 只使用点坐标

    # 转换为 PyTorch 张量
    data = torch.tensor(data, dtype=torch.float32)
    # 将数据维度扩展到 [B, C, N]，即 [batch_size, 3, num_points]
    data = data.unsqueeze(0).transpose(2, 1)  # 转换为 [1, 3, N]
    return data


# 3. 使用模型进行预测并输出每个类别的概率
def predict(model, point_cloud_data):
    with torch.no_grad():  # 关闭梯度计算
        output, _ = model(point_cloud_data)
        # 计算每个类别的概率（应用 softmax）
        probabilities = torch.nn.functional.softmax(output, dim=1)  # 对输出进行 softmax，dim=1 是对类别维度应用
        predicted_class = torch.argmax(probabilities, 1)  # 获取最大概率的类别
    return predicted_class.item(), probabilities


# 4. 加载模型和点云数据并进行预测
if __name__ == "__main__":
    model_path = "./checkpoints/best_model.pth"  # 替换为你的权重文件路径
    file_path = "processed_points.txt"  # 替换为你的点云数据路径
    num_class = 10  # 替换为你的类别数

    # 加载模型
    model = load_model(model_path, num_class)

    # 加载点云数据
    point_cloud_data = load_point_cloud(file_path)

    # 进行分类预测
    predicted_class, probabilities = predict(model, point_cloud_data)

    # 输出预测结果
    print(f"Predicted class: {predicted_class}")
    print("Class probabilities:")
    for i, prob in enumerate(probabilities[0]):  # probabilities[0] 是 batch size 为 1 的情况下的结果
        print(f"Class {i}: {prob.item():.4f}")
