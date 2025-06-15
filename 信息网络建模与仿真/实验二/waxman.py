import random
import math
import networkx as nx
import matplotlib.pyplot as plt

class WaxmanModel:
    def __init__(self, n, alpha, beta, seed=None):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.nodes = []
        self.edges = []

    def generate_topology(self):
        # 设置随机种子
        if self.seed is not None:
            random.seed(self.seed)
        # 随机生成节点坐标
        self.nodes = [(random.random(), random.random()) for _ in range(self.n)]
        # 计算平面的对角线长度
        L = math.sqrt(2)
        # 生成边
        self.edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # 计算欧几里得距离
                d = math.sqrt((self.nodes[i][0] - self.nodes[j][0])**2 + (self.nodes[i][1] - self.nodes[j][1])**2)
                # 计算连接概率
                p = self.alpha * math.exp(-d / (self.beta * L))
                if random.random() < p:
                    self.edges.append((i, j))
        return self.nodes, self.edges

# 示例用法
waxman = WaxmanModel(n=100, alpha=0.5, beta=0.1, seed=42)
nodes, edges = waxman.generate_topology()

# 绘制网络拓扑图
G = nx.Graph()
G.add_nodes_from(range(len(nodes)))
G.add_edges_from(edges)
nx.draw(G, pos=dict(enumerate(nodes)), with_labels=True, node_size=50)
plt.title("Waxman拓扑图")
plt.show()

# 打印验证信息
print(f"生成的Waxman拓扑包含{len(nodes)}个节点和{len(edges)}条边")
