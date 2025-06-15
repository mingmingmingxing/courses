import random
import networkx as nx
import matplotlib.pyplot as plt

class BaModel:
    def __init__(self, n, m, seed=None):
        self.n = n
        self.m = m
        self.seed = seed
        self.graph = {}

    def generate_topology(self):
        # 设置随机种子
        if self.seed is not None:
            random.seed(self.seed)
        # 初始网络为一个环状图（m+1个节点组成的环）
        self.graph = {i: [] for i in range(self.m + 1)}
        for i in range(self.m + 1):
            self.graph[i].append((i + 1) % (self.m + 1))
            self.graph[(i + 1) % (self.m + 1)].append(i)
        # 逐步添加剩余节点
        for new_node in range(self.m + 1, self.n):
            self.graph[new_node] = []
            # 选择m个现有节点进行连接
            for _ in range(self.m):
                # 计算各现有节点的度，得到度分布列表
                degrees = [len(neighbors) for neighbors in self.graph.values()]
                total_degree = sum(degrees)
                # 根据度分布列表，按概率选择一个现有节点k
                probabilities = [degree / total_degree for degree in degrees]
                # 选择一个节点索引
                target_node = random.choices(list(self.graph.keys()), probabilities)[0]
                # 添加边（双向连接）
                self.graph[new_node].append(target_node)
                self.graph[target_node].append(new_node)
        return self.graph

# 示例用法
ba = BaModel(n=100, m=3, seed=42)
graph = ba.generate_topology()

# 绘制网络拓扑图
G = nx.Graph()
for node in graph:
    G.add_edges_from([(node, neighbor) for neighbor in graph[node]])
nx.draw(G, with_labels=True, node_size=50)
plt.title("BA拓扑图")
plt.show()

# 打印验证信息
print(f"生成的BA拓扑包含{len(graph)}个节点")
total_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
print(f"总边数：{total_edges}")