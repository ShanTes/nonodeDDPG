# 导入依赖库
import os
import sys
import copy
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 自定义库导入
sys.path.append(r"D:\BASE\AoI-V2X-IEEE-TVT-2023-main\AoI-V2X-IEEE-TVT-2023-main\4-DDPG")
from Classes.buffer import ReplayBuffer, PrioritizedReplayBuffer
from ddpg_torch import DDPG
from Classes.Environment_Platoon import Environ
import scipy.io as sio


def train():
    """DDPG训练主函数 - 优化版本"""
    
    # 初始化仿真环境
    env1 = Environ()
    
    # 获取环境参数
    n_vehicle = env1.n_Vehicle
    n_platoon = env1.n_platoon
    n_RB = env1.n_RB
    
    # 计算状态和动作空间维度
    state_dim = n_vehicle * n_RB + n_vehicle + n_platoon
    action_dim = n_vehicle * n_RB
    
    print(f"环境配置: 车辆={n_vehicle}, 编队={n_platoon}, 资源块={n_RB}")
    print(f"状态维度={state_dim}, 动作维度={action_dim}")
    
    # 超参数配置
    discount = 0.9
    tau = 0.01
    
    actor_lr = 1e-4
    critic_lr = 1e-3
    
    use_prioritized_replay = True
    buffer_size = int(1e6)
    batch_size = 64
    
    episodes = 500
    warmup_steps = 1000
    train_freq = 2
    eval_freq = 10
    
    noise_initial = 0.5
    noise_final = 0.05
    noise_decay = 0.995
    
    # 创建保存目录
    save_dir = r"D:\BASE\AoI-V2X-IEEE-TVT-2023-main\AoI-V2X-IEEE-TVT-2023-main\4-DDPG\model\marl_model"
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化Tensorboard
    log_dir = os.path.join(save_dir, "tensorboard_logs")
    writer = SummaryWriter(log_dir)
    
    # 初始化DDPG智能体
    ddpg_agent = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        discount=discount,
        tau=tau,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        use_prioritized_replay=use_prioritized_replay
    )
    
    # 初始化经验回放缓冲区
    if use_prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=buffer_size
        )
    else:
        replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=buffer_size
        )
    
    # 训练过程
    training_rewards = []
    eval_rewards = []
    best_reward = -np.inf
    best_eval_reward = -np.inf
    total_steps = 0
    noise = noise_initial
    
    print("\n开始训练DDPG智能体")
    
    for episode in tqdm(range(episodes), desc="训练进度"):
        env1.random_seed = episode
        env1.new_random_game()
        env1.compute_channel()
        state = env1.get_state()
        
        episode_reward = 0
        episode_steps = 0
        done = False
        
        noise = max(noise_final, noise * noise_decay)
        
        while not done:
            action = ddpg_agent.select_action(state, noise)
            action_reshaped = action.reshape(n_vehicle, n_RB)
            next_state, reward, done = env1.step(action_reshaped)
            
            replay_buffer.add(state, action, next_state, reward, float(done))
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if total_steps > warmup_steps and total_steps % train_freq == 0:
                actor_loss, critic_loss = ddpg_agent.train(replay_buffer, batch_size)
                writer.add_scalar('Loss/Actor', actor_loss, total_steps)
                writer.add_scalar('Loss/Critic', critic_loss, total_steps)
        
        training_rewards.append(episode_reward)
        
        writer.add_scalar('Reward/Episode', episode_reward, episode)
        writer.add_scalar('Exploration/Noise', noise, episode)
        
        lr_info = ddpg_agent.get_lr()
        writer.add_scalar('LR/Actor', lr_info['actor_lr'], episode)
        writer.add_scalar('LR/Critic', lr_info['critic_lr'], episode)
        
        ddpg_agent.step_scheduler()
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            ddpg_agent.save(os.path.join(save_dir, "best_model.pth"))
        
        if (episode + 1) % eval_freq == 0:
            eval_reward = evaluate(ddpg_agent, env1, n_vehicle, n_RB)
            eval_rewards.append(eval_reward)
            writer.add_scalar('Reward/Eval_Mean', eval_reward, episode)
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                ddpg_agent.save(os.path.join(save_dir, "best_eval_model.pth"))
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(training_rewards[-10:])
            print(f"\nEpisode {episode+1}: 平均奖励={avg_reward:.4f}, 噪声={noise:.4f}")
    
    # 保存结果
    ddpg_agent.save(os.path.join(save_dir, "final_model.pth"))
    sio.savemat(os.path.join(save_dir, "reward.mat"), {"reward": np.array(training_rewards)})
    sio.savemat(os.path.join(save_dir, "critic_loss.mat"), {"critic_loss": np.array(ddpg_agent.critic_losses)})
    writer.close()
    
    print(f"\n训练完成! 最佳训练奖励={best_reward:.4f}, 最佳评估奖励={best_eval_reward:.4f}")
    return training_rewards, eval_rewards


def evaluate(agent, env, n_vehicle, n_RB, eval_episodes=5):
    """评估函数"""
    eval_rewards = []
    
    for _ in range(eval_episodes):
        env.new_random_game()
        env.compute_channel()
        state = env.get_state()
        
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, noise=0.0)
            action_reshaped = action.reshape(n_vehicle, n_RB)
            next_state, reward, done = env.step(action_reshaped)
            state = next_state
            episode_reward += reward
        
        eval_rewards.append(episode_reward)
    
    return np.mean(eval_rewards)


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    training_rewards, eval_rewards = train()