import pickle
import numpy as np
import os

source = "cheetah_vel"
def update_trajectory_rewards(trajectory, original_goal_vel, new_goal_vel):
    updated_trajectory = []

    for step in trajectory:
        actions = step['actions']
        old_rewards = step['rewards'] 

        new_rewards = []
        for t in range(len(old_rewards)):
            old_reward = old_rewards[t, 0] 

            ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(actions[t])) 

            forward_reward = old_reward + ctrl_cost

            forward_vel = abs(-forward_reward + original_goal_vel)

            new_forward_reward = -1.0 * abs(forward_vel - new_goal_vel)

            new_reward = new_forward_reward - ctrl_cost

            new_rewards.append([new_reward])

        updated_step = step.copy()
        updated_step['rewards'] = np.array(new_rewards, dtype=np.float32)
        updated_trajectory.append(updated_step)

    return updated_trajectory

def process_trajectories(base_path, config_path, new_config_path):
    test_set = [13, 16, 24, 32, 37]
    for i in test_set:
        trajectory_file = os.path.join(base_path, f"cheetah_vel-{i}-expert.pkl")
        with open(trajectory_file, 'rb') as f:
            trajectory = pickle.load(f)

        original_config_file = os.path.join(config_path, f"config_cheetah_vel_task{i}.pkl")
        with open(original_config_file, 'rb') as f:
            original_config = pickle.load(f)
        original_goal_vel = original_config[0]['velocity']

        new_config_file = os.path.join(new_config_path, f"config_{source}_task{i}.pkl")
        with open(new_config_file, 'rb') as f:
            new_config = pickle.load(f)
        new_goal_vel = 3 - new_config[0]['velocity']
        new_config[0]['velocity'] = 3 - new_config[0]['velocity']

        with open(new_config_file, 'wb') as f:
            pickle.dump(new_config, f)
        print(f"Updated config {i} saved as {new_config_file}")
        updated_trajectory = update_trajectory_rewards(trajectory, original_goal_vel, new_goal_vel)
        
        output_file = os.path.join(output_path, f"{source}-{i}-expert.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(updated_trajectory, f)

        
        print(f"Updated trajectory {i} saved as {output_file}")

base_path = "/home/yaningg/workspace/prompt-dt-main/data/cheetah_vel"
config_path = "/home/yaningg/workspace/prompt-dt-main/config/cheetah_vel"
new_config_path = f"/home/yaningg/workspace/prompt-dt-main/config/{source}"
output_path = f"/home/yaningg/workspace/prompt-dt-main/data/{source}"

process_trajectories(base_path, config_path, new_config_path)
