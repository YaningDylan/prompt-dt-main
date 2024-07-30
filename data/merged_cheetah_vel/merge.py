import os
import pickle
import numpy as np

def load_trajectory(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_trajectory(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def merge_trajectories(traj1, traj2):
    merged_trajectories = []
    
    for t1, t2 in zip(traj1, traj2):
        merged_trajectory = {}
        
        for key in t1.keys():
            data1 = t1[key]
            data2 = t2[key]

            if key == 'rewards':
                # 对于rewards，取平均值
                merged_data = (data1 + data2) / 2
            else:
                # 对于其他数据，在每个时间步随机选择data1或data2
                mask = np.random.rand(len(data1), 1) > 0.5
                merged_data = np.where(mask, data1, data2)

            merged_trajectory[key] = merged_data

        merged_trajectories.append(merged_trajectory)

    return merged_trajectories

def process_total_data_mean(total_traj_list, mode):
    states = []
    for traj in total_traj_list:
        if isinstance(traj, dict) and 'observations' in traj:
            states.append(traj['observations'])
        elif isinstance(traj, list):
            for path in traj:
                if 'observations' in path:
                    states.append(path['observations'])
                else:
                    raise ValueError(f"Unexpected structure in trajectory: {path}")
        else:
            raise ValueError(f"Unexpected structure in trajectory list: {traj}")
    
    states = np.concatenate(states, axis=0)
    state_mean = np.mean(states, axis=0)
    state_std = np.std(states, axis=0)
    return state_mean, state_std

def main():
    base_path = '/root/prompt-dt-main/data/cheetah_vel/'
    output_path = '/root/prompt-dt-main/data/merged_cheetah_vel/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    total_traj_list = []

    for i in range(0, 40, 2):
        file_path1 = os.path.join(base_path, f'cheetah_vel-{i}-prompt-expert.pkl')
        file_path2 = os.path.join(base_path, f'cheetah_vel-{i+1}-prompt-expert.pkl')
        
        if not os.path.exists(file_path1):
            print(f"File {file_path1} not found.")
            continue
        if not os.path.exists(file_path2):
            print(f"File {file_path2} not found.")
            continue

        traj1 = load_trajectory(file_path1)
        traj2 = load_trajectory(file_path2)
        
        merged_traj = merge_trajectories(traj1, traj2)
        total_traj_list.extend(merged_traj)
        
        output_file = os.path.join(output_path, f'merged_cheetah_vel-{int(i/2)}-prompt-expert.pkl')
        save_trajectory(merged_traj, output_file)
        print(f'Merged trajectories saved to {output_file}')

    print(total_traj_list[:2])
    total_state_mean, total_state_std = process_total_data_mean(total_traj_list, mode='normal')
    print(f"Total state mean: {total_state_mean}, Total state std: {total_state_std}")

if __name__ == '__main__':
    main()
