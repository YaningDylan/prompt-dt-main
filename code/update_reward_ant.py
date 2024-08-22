import pickle
import numpy as np
import os

source = "ant_dir"

def update_trajectory_rewards(trajectory, original_dir, new_dir):
    updated_trajectory = []

    for step in trajectory:
        obs = step['observations']
        old_rewards = step['rewards']  
        new_rewards = []
        for t in range(len(old_rewards)):
            x_vel = obs[t, 13]  
            y_vel = obs[t, 14]  
            z_vel = obs[t, 15]  
            dt = 0.05
            torso_velocity = np.array([x_vel, y_vel, z_vel])
            old_forward_reward = np.dot((torso_velocity[:2]/dt), original_dir)
            old_reward = old_rewards[t, 0]
            new_forward_reward = np.dot((torso_velocity[:2]/dt), new_dir)
            new_reward = old_reward + new_forward_reward - old_forward_reward

            new_rewards.append([new_reward])

        updated_step = step.copy()
        updated_step['rewards'] = np.array(new_rewards, dtype=np.float32)
        updated_trajectory.append(updated_step)

    return updated_trajectory

def process_trajectories(base_path, config_path, new_config_path):
    test_set = [1,4,41]
    for i in test_set:
        trajectory_file = os.path.join(base_path, f"ant_dir-{i}-expert.pkl")
        with open(trajectory_file, 'rb') as f:
            trajectory = pickle.load(f)

        original_config_file = os.path.join(config_path, f"config_ant_dir_task{i}.pkl")
        with open(original_config_file, 'rb') as f:
            original_config = pickle.load(f)
        original_dir = original_config[0]['goal']
        old_direct = (np.cos(original_dir), np.sin(original_dir))
        new_config_file = os.path.join(new_config_path, f"config_{source}_task{i}.pkl")
        with open(new_config_file, 'rb') as f:
            new_config = pickle.load(f)
        new_dir = 2*np.pi - new_config[0]['goal']
        new_direct = (np.cos(new_dir), np.sin(new_dir))
        
        new_config[0]['goal'] = new_dir

        with open(new_config_file, 'wb') as f:
            pickle.dump(new_config, f)
        print(f"Updated config {i} saved as {new_config_file}")
        updated_trajectory = update_trajectory_rewards(trajectory, old_direct, new_direct)

        output_file = os.path.join(output_path, f"{source}-{i}-expert.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(updated_trajectory, f)

        print(f"Updated trajectory {i} saved as {output_file}")

base_path = "/home/yaningg/workspace/prompt-dt-main/data/ant_dir"
config_path = "/home/yaningg/workspace/prompt-dt-main/config/ant_dir"
new_config_path = f"/home/yaningg/workspace/prompt-dt-main/config/{source}"
output_path = f"/home/yaningg/workspace/prompt-dt-main/data/{source}"

process_trajectories(base_path, config_path, new_config_path)
