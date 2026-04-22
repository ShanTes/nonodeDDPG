import numpy as np
import torch


class ReplayBuffer:
    """标准经验回放缓冲区"""
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def add(self, state, action, next_state, reward, done):
        """添加经验到缓冲区"""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """随机采样一批经验"""
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区 - 优化版本
    根据TD误差的大小来采样，重要经验被采样的概率更高
    """
    def __init__(self, state_dim, action_dim, max_size=int(1e6), alpha=0.6, beta=0.4, beta_increment=0.001):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.alpha = alpha  # 优先级的指数
        self.beta = beta    # 重要性采样的权重
        self.beta_increment = beta_increment  # beta逐渐增加到1
        self.epsilon = 1e-6  # 防止优先级为0
        
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        
        # 优先级存储
        self.priorities = np.zeros((max_size,))
        self.max_priority = 1.0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def add(self, state, action, next_state, reward, done):
        """添加经验到缓冲区，新经验使用最大优先级"""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        
        # 新经验使用最大优先级
        self.priorities[self.ptr] = self.max_priority
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """根据优先级采样一批经验"""
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # 根据概率采样索引
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        
        # 计算重要性采样权重
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化权重
        weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
        
        return (
            torch.FloatTensor(self.state[indices]).to(self.device),
            torch.FloatTensor(self.action[indices]).to(self.device),
            torch.FloatTensor(self.next_state[indices]).to(self.device),
            torch.FloatTensor(self.reward[indices]).to(self.device),
            torch.FloatTensor(self.not_done[indices]).to(self.device),
            indices,
            weights
        )
    
    def update_priorities(self, indices, td_errors):
        """根据TD误差更新优先级"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)