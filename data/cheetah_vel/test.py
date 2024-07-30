import pickle
import numpy as np

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print(data)
    return data


def inspect_trajectory(data):
    if isinstance(data, list):
        for i, traj in enumerate(data):
            print(f"Trajectory {i}:")
            if isinstance(traj, dict):
                for key, value in traj.items():
                    print(f"  {key}: {type(value)}, shape: {getattr(value, 'shape', 'N/A')}")
            else:
                print(f"  Data type: {type(traj)}")
    elif isinstance(data, dict):
        for key, value in data.items():
            print(f"{key}: {type(value)}, shape: {getattr(value, 'shape', 'N/A')}")
    else:
        print(f"Data type: {type(data)}")

# 修改文件路径为你的pkl文件路径
file_path = '/root/prompt-dt-main/data/cheetah_vel/cheetah_vel-0-prompt-expert.pkl'
data = load_pkl(file_path)
print(data)

# 查看数据结构
inspect_trajectory(data)
