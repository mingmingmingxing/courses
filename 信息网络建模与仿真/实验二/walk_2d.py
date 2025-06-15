import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class RandomWalk2dMobilityModel:
    def __init__(self, n_nodes, bounds=(-100, 100, -100, 100), mode='time', mode_param=1.0,
                 speed_range=(0.0, 10.0), direction_range=(0.0, 2*np.pi)):
        self.n_nodes = n_nodes
        self.bounds = bounds
        self.mode = mode
        self.mode_param = mode_param
        self.speed_range = speed_range
        self.direction_range = direction_range
        self.positions = np.random.uniform(low=bounds[0], high=bounds[1], size=(n_nodes, 2))
        self.speeds = np.random.uniform(low=speed_range[0], high=speed_range[1], size=n_nodes)
        self.directions = np.random.uniform(low=direction_range[0], high=direction_range[1], size=n_nodes)
        self.trails = [[] for _ in range(n_nodes)]
        for i in range(n_nodes):
            self.trails[i].append((self.positions[i, 0], self.positions[i, 1]))

    def update(self):
        # 根据模式更新位置
        if self.mode == 'time':
            dx = self.speeds * np.cos(self.directions) * self.mode_param
            dy = self.speeds * np.sin(self.directions) * self.mode_param
        elif self.mode == 'distance':
            dx = self.speeds * np.cos(self.directions)
            dy = self.speeds * np.sin(self.directions)
            # 计算移动步数
            steps = int(self.mode_param / np.mean(self.speeds))
            dx *= steps
            dy *= steps
        else:
            raise ValueError("Mode must be 'time' or 'distance'")

        new_positions = self.positions + np.stack((dx, dy), axis=-1)

        # 检查边界并反弹
        for i in range(self.n_nodes):
            if new_positions[i, 0] < self.bounds[0] or new_positions[i, 0] > self.bounds[1]:
                self.directions[i] = np.pi - self.directions[i]
            if new_positions[i, 1] < self.bounds[2] or new_positions[i, 1] > self.bounds[3]:
                self.directions[i] = -self.directions[i]
            # 限制在边界内
            new_positions[i, 0] = np.clip(new_positions[i, 0], self.bounds[0], self.bounds[1])
            new_positions[i, 1] = np.clip(new_positions[i, 1], self.bounds[2], self.bounds[3])

        # 更新状态
        self.positions = new_positions

        # 记录轨迹
        for i in range(self.n_nodes):
            self.trails[i].append((self.positions[i, 0], self.positions[i, 1]))

    def get_trails(self):
        return self.trails

# 示例用法
model = RandomWalk2dMobilityModel(n_nodes=5, bounds=(-100, 100, -100, 100), mode='time', mode_param=1.0)
for _ in range(100):
    model.update()

trails = model.get_trails()
for trail in trails:
    xs, ys = zip(*trail)
    plt.plot(xs, ys)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Random Walk 2D Mobility Model Trails')
plt.show()