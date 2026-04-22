import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    """Actor网络（策略网络）- 优化版本"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        
        # 网络层定义
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)  # 使用LayerNorm替代BatchNorm，更稳定
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.LayerNorm(hidden_dim // 2)
        
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)
        
        # 权重初始化
        self._init_weights()
        
    def _init_weights(self):
        """He初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        # 使用tanh将输出限制在[-1, 1]
        action = torch.tanh(self.fc4(x))
        return action


class Critic(nn.Module):
    """Critic网络（价值网络）- 优化版本"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1网络
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.bn1_q1 = nn.LayerNorm(hidden_dim)
        
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2_q1 = nn.LayerNorm(hidden_dim)
        
        self.fc3_q1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3_q1 = nn.LayerNorm(hidden_dim // 2)
        
        self.fc4_q1 = nn.Linear(hidden_dim // 2, 1)
        
        # Q2网络（Double Q-learning减少过估计）
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.bn1_q2 = nn.LayerNorm(hidden_dim)
        
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2_q2 = nn.LayerNorm(hidden_dim)
        
        self.fc3_q2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3_q2 = nn.LayerNorm(hidden_dim // 2)
        
        self.fc4_q2 = nn.Linear(hidden_dim // 2, 1)
        
        # 权重初始化
        self._init_weights()
        
    def _init_weights(self):
        """He初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)
        
        # Q1前向传播
        x1 = F.relu(self.bn1_q1(self.fc1_q1(xu)))
        x1 = F.relu(self.bn2_q1(self.fc2_q1(x1)))
        x1 = F.relu(self.bn3_q1(self.fc3_q1(x1)))
        q1 = self.fc4_q1(x1)
        
        # Q2前向传播
        x2 = F.relu(self.bn1_q2(self.fc1_q2(xu)))
        x2 = F.relu(self.bn2_q2(self.fc2_q2(x2)))
        x2 = F.relu(self.bn3_q2(self.fc3_q2(x2)))
        q2 = self.fc4_q2(x2)
        
        return q1, q2
    
    def Q1(self, state, action):
        """只返回Q1的值，用于Actor更新"""
        xu = torch.cat([state, action], dim=1)
        
        x1 = F.relu(self.bn1_q1(self.fc1_q1(xu)))
        x1 = F.relu(self.bn2_q1(self.fc2_q1(x1)))
        x1 = F.relu(self.bn3_q1(self.fc3_q1(x1)))
        q1 = self.fc4_q1(x1)
        
        return q1