import pickle

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# 加载pkl文件
file_path = '/root/prompt-dt-main/config/cheetah_vel/config_cheetah_vel_task0.pkl'  # 替换为你的pkl文件路径
data = load_pkl(file_path)
print(data)

# 查看数据结构
print(type(data))  # 打印数据的类型
if isinstance(data, dict):
    for key, value in data.items():
        print(f"{key}: {type(value)}, shape: {getattr(value, 'shape', 'N/A')}")
elif isinstance(data, list):
    print(f"List of length {len(data)}")
    for i, item in enumerate(data):
        if isinstance(item, dict):
            for key, value in item.items():
                print(f"{key}: {type(value)}, shape: {getattr(value, 'shape', 'N/A')}")
        print(f"Item {i}: {type(item)}, shape: {getattr(item, 'shape', 'N/A')}")
else:
    print("Data is of type:", type(data))
