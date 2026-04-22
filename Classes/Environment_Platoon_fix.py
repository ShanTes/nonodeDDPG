"""
Environment_Platoon.py 缺失方法的修复
这些方法需要添加到 Environ 类中
"""

import numpy as np

def compute_channel(self, number_vehicle=None, size_platoon=None):
    """计算信道 - 兼容 Main.py 的调用"""
    if number_vehicle is None:
        number_vehicle = self.n_Veh
    if size_platoon is None:
        size_platoon = self.size_platoon
    return self.renew_channel(number_vehicle, size_platoon)


def get_state(self):
    """
    获取当前环境状态
    
    返回: 包含以下信息的state向量
    - 所有车辆对所有RB的信道增益 (n_vehicle * n_RB)
    - 所有车辆的AoI (n_vehicle)
    - 编队层面的需求/状态 (n_platoon)
    """
    n_platoon = self.n_Veh // self.size_platoon
    
    # 1. 提取信道增益信息 (V2I channel gains for each platoon)
    # 使用对数形式的信道增益，归一化到 [0, 1]
    if hasattr(self, 'V2I_channels_with_fastfading'):
        channel_gain = -self.V2I_channels_with_fastfading[:, :, 0]  # 取第一个RB作为代表
        # 归一化到 [0, 1]
        channel_gain = (channel_gain - channel_gain.min()) / (channel_gain.max() - channel_gain.min() + 1e-8)
        state_channels = channel_gain.flatten()[:self.n_RB * n_platoon]
    else:
        state_channels = np.zeros(self.n_RB * n_platoon)
    
    # 2. AoI信息，归一化
    if hasattr(self, 'AoI'):
        aoi_normalized = self.AoI / 100.0  # 归一化AoI
    else:
        aoi_normalized = np.zeros(n_platoon)
    
    # 3. V2V需求比例
    if hasattr(self, 'V2V_demand'):
        demand_ratio = self.V2V_demand / self.V2V_demand_size
    else:
        demand_ratio = np.zeros(n_platoon)
    
    # 拼接所有状态信息
    state = np.concatenate([
        state_channels,
        aoi_normalized,
        demand_ratio
    ])
    
    return state


def step(self, action):
    """
    执行一步仿真
    
    参数:
        action: 动作矩阵 [n_platoon, 3]
               - action[:, 0]: 选择的RB (0 到 n_RB-1)
               - action[:, 1]: 编队决策 (0=V2I通信, 1=V2V通信)
               - action[:, 2]: 发射功率 (dBm, 范围 [0, 30])
    
    返回:
        next_state: 下一个状态
        reward: 全局奖励
        done: 是否结束 (所有V2V需求都满足)
    """
    n_platoon = self.n_Veh // self.size_platoon
    
    # 1. 确保action形状正确
    if action.ndim == 1:
        action = action.reshape(n_platoon, 3)
    
    # 2. 将动作从 [-1, 1] 映射到实际范围
    # RB选择: [0, n_RB-1]
    action_scaled = action.copy()
    action_scaled[:, 0] = ((action[:, 0] + 1) / 2 * self.n_RB).astype(int)
    action_scaled[:, 0] = np.clip(action_scaled[:, 0], 0, self.n_RB - 1)
    
    # 编队决策: 0 或 1
    action_scaled[:, 1] = (action[:, 1] > 0).astype(int)
    
    # 发射功率: [0, 30] dBm
    action_scaled[:, 2] = ((action[:, 2] + 1) / 2 * 30)
    action_scaled[:, 2] = np.clip(action_scaled[:, 2], 0, 30)
    
    # 3. 更新快衰落信道
    self.renew_channels_fastfading()
    
    # 4. 执行动作并获取奖励
    per_user_reward, global_reward, platoon_AoI, C_rate, V_rate, Demand, V2V_success = \
        self.act_for_training(action_scaled)
    
    # 5. 更新车辆位置 (每100个快衰更新一次位置和慢衰)
    if not hasattr(self, '_step_count'):
        self._step_count = 0
    self._step_count += 1
    
    if self._step_count >= (self.time_slow / self.time_fast):
        self.renew_positions()
        self.renew_channel(self.n_Veh, self.size_platoon)
        self._step_count = 0
    
    # 6. 获取下一个状态
    next_state = self.get_state()
    
    # 7. 判断episode是否结束
    # 当所有V2V需求都满足时结束
    if np.all(Demand <= 0):
        done = True
    # 或者超过最大步数
    elif hasattr(self, '_episode_steps'):
        self._episode_steps += 1
        done = self._episode_steps >= 1000
    else:
        self._episode_steps = 1
        done = False
    
    return next_state, global_reward, done


# 将这些方法添加到Environ类中
import sys
if __name__ != '__main__':
    # 动态添加方法到Environ类
    from Classes.Environment_Platoon import Environ
    Environ.compute_channel = compute_channel
    Environ.get_state = get_state
    Environ.step = step
