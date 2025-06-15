import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class GaussMarkovMobilityModel:
    def __init__(self, n_nodes, alpha=0.85, bounds=(-100, 100, -100, 100), time_step=1.0,
                 mean_velocity=(0.0, 10.0), mean_direction=(0.0, 2*np.pi), mean_pitch=(0.0, 0.0)):
        self.n_nodes = n_nodes
        self.alpha = alpha
        self.bounds = bounds
        self.time_step = time_step
        self.mean_velocity = mean_velocity
        self.mean_direction = mean_direction
        self.mean_pitch = mean_pitch
        self.positions = np.random.uniform(low=bounds[0], high=bounds[1], size=(n_nodes, 2))
        self.velocities = np.random.uniform(low=mean_velocity[0], high=mean_velocity[1], size=n_nodes)
        self.directions = np.random.uniform(low=mean_direction[0], high=mean_direction[1], size=n_nodes)
        self.pitches = np.random.uniform(low=mean_pitch[0], high=mean_pitch[1], size=n_nodes)
        self.trails = [[] for _ in range(n_nodes)]
        for i in range(n_nodes):
            self.trails[i].append((self.positions[i, 0], self.positions[i, 1]))

    def update(self):
        # 更新速度、方向和俯仰角
        new_velocities = self.alpha * self.velocities + (1 - self.alpha) * np.random.uniform(low=self.mean_velocity[0], high=self.mean_velocity[1], size=self.n_nodes)
        new_directions = self.alpha * self.directions + (1 - self.alpha) * np.random.uniform(low=self.mean_direction[0], high=self.mean_direction[1], size=self.n_nodes)
        new_pitches = self.alpha * self.pitches + (1 - self.alpha) * np.random.uniform(low=self.mean_pitch[0], high=self.mean_pitch[1], size=self.n_nodes)

        # 计算新的位置
        dx = new_velocities * np.cos(new_directions) * self.time_step
        dy = new_velocities * np.sin(new_directions) * self.time_step
        new_positions = self.positions + np.stack((dx, dy), axis=-1)

        # 检查边界并反弹
        for i in range(self.n_nodes):
            if new_positions[i, 0] < self.bounds[0] or new_positions[i, 0] > self.bounds[1]:
                new_directions[i] = np.pi - new_directions[i]
            if new_positions[i, 1] < self.bounds[2] or new_positions[i, 1] > self.bounds[3]:
                new_directions[i] = -new_directions[i]
            # 限制在边界内
            new_positions[i, 0] = np.clip(new_positions[i, 0], self.bounds[0], self.bounds[1])
            new_positions[i, 1] = np.clip(new_positions[i, 1], self.bounds[2], self.bounds[3])

        # 更新状态
        self.velocities = new_velocities
        self.directions = new_directions
        self.pitches = new_pitches
        self.positions = new_positions

        # 记录轨迹
        for i in range(self.n_nodes):
            self.trails[i].append((self.positions[i, 0], self.positions[i, 1]))

    def get_trails(self):
        return self.trails

# 示例用法
model = GaussMarkovMobilityModel(n_nodes=5, alpha=0.85, bounds=(-100, 100, -100, 100), time_step=1.0)
for _ in range(100):
    model.update()

trails = model.get_trails()
for trail in trails:
    xs, ys = zip(*trail)
    plt.plot(xs, ys)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Gauss-Markov Mobility Model Trails')
plt.show()