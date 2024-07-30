import pickle
import os

def load_config(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_config(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def merge_configs(config1, config2):
    merged_config = []
    
    for c1, c2 in zip(config1, config2):
        merged_item = {
            'velocity': (c1['velocity'] + c2['velocity']) / 2  # 对于velocity，取平均值
        }
        merged_config.append(merged_item)

    return merged_config

def main():
    base_path = '/root/prompt-dt-main/config/cheetah_vel/'
    output_path = '/root/prompt-dt-main/config/merged_cheetah_vel/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in range(0, 40, 2):
        file_path1 = os.path.join(base_path, f'config_cheetah_vel_task{i}.pkl')
        file_path2 = os.path.join(base_path, f'config_cheetah_vel_task{i+1}.pkl')
        
        config1 = load_config(file_path1)
        config2 = load_config(file_path2)
        
        merged_config = merge_configs(config1, config2)
        
        output_file = os.path.join(output_path, f'config_merged_cheetah_vel_task{int(i/2)}.pkl')
        save_config(merged_config, output_file)
        print(f'Merged config saved to {output_file}')

if __name__ == '__main__':
    main()
