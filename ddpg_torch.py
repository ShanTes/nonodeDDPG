import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Classes import networks, buffer


class DDPG(object):
    """DDPG算法实现 - 优化版本
    主要优化：
    1. 支持Double Q-learning减少过估计
    2. 支持优先经验回放
    3. 添加梯度裁剪
    4. 添加学习率调度
    """
    def __init__(self, state_dim, action_dim, discount=0.99, tau=0.005, 
                 actor_lr=1e-4, critic_lr=1e-3, use_prioritized_replay=False):
        
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 初始化Actor网络和目标网络
        self.actor = networks.Actor(state_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # 初始化Critic网络和目标网络
        self.critic = networks.Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 学习率调度器
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(
            self.actor_optimizer, step_size=100, gamma=0.95
        )
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(
            self.critic_optimizer, step_size=100, gamma=0.95
        )
        
        # 超参数
        self.discount = discount
        self.tau = tau
        self.use_prioritized_replay = use_prioritized_replay
        
        # 梯度裁剪阈值
        self.max_grad_norm = 1.0
        
        # 训练统计
        self.actor_losses = []
        self.critic_losses = []
    
    def select_action(self, state, noise=0.0):
        """选择动作（可添加噪声进行探索）"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        # 添加探索噪声
        if noise > 0:
            action = action + np.random.normal(0, noise, size=action.shape)
        
        # 将动作限制在[-1, 1]范围内
        action = np.clip(action, -1, 1)
        
        return action
    
    def train(self, replay_buffer, batch_size=256):
        """训练DDPG智能体 - 支持优先经验回放"""
        
        if self.use_prioritized_replay:
            # 优先经验回放采样
            state, action, next_state, reward, not_done, indices, weights = \
                replay_buffer.sample(batch_size)
        else:
            # 标准经验回放采样
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            weights = torch.ones_like(reward)
        
        # ==================== 训练Critic ====================
        with torch.no_grad():
            # 计算目标Q值（使用Double Q-learning）
            next_action = self.actor_target(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)  # 取较小值减少过估计
            target_Q = reward + not_done * self.discount * target_Q
        
        # 计算当前Q值
        current_Q1, current_Q2 = self.critic(state, action)
        
        # 计算Critic损失（Double Q-learning）
        critic_loss = F.mse_loss(current_Q1, target_Q, reduction='none') + \
                      F.mse_loss(current_Q2, target_Q, reduction='none')
        
        # 应用重要性采样权重
        critic_loss = (critic_loss * weights).mean()
        
        # 优化Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # ==================== 训练Actor ====================
        # 计算Actor损失
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        
        # 优化Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # ==================== 软更新目标网络 ====================
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        # 如果使用优先经验回放，更新优先级
        if self.use_prioritized_replay:
            with torch.no_grad():
                # 计算TD误差用于更新优先级
                td_errors = torch.abs(current_Q1 - target_Q).cpu().numpy()
                replay_buffer.update_priorities(indices, td_errors)
        
        # 记录损失
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        
        return actor_loss.item(), critic_loss.item()
    
    def _soft_update(self, network, target_network):
        """软更新目标网络参数"""
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def step_scheduler(self):
        """更新学习率调度器"""
        self.actor_scheduler.step()
        self.critic_scheduler.step()
    
    def get_lr(self):
        """获取当前学习率"""
        return {
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr']
        }
    
    def save(self, filename):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'critic_scheduler': self.critic_scheduler.state_dict(),
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
        }, filename)
        print(f"模型已保存到: {filename}")
    
    def load(self, filename):
        """加载模型"""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler'])
        
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler'])
        
        self.actor_losses = checkpoint.get('actor_losses', [])
        self.critic_losses = checkpoint.get('critic_losses', [])
        
        print(f"模型已从 {filename} 加载")