import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

#DATA ADT

class Building:
    def __init__(self, id, name, location):
        self.BuildingID = id
        self.BuildingName = name
        self.LocationDetails = location
        self.Connections = []


#BINARY SEARCH TREE

class BSTNode:
    def __init__(self, building):
        self.data = building
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, node, building):
        if node is None:
            return BSTNode(building)
        if building.BuildingID < node.data.BuildingID:
            node.left = self.insert(node.left, building)
        else:
            node.right = self.insert(node.right, building)
        return node

    def inorder(self, node, result):
        if node:
            self.inorder(node.left, result)
            result.append(node.data.BuildingName)
            self.inorder(node.right, result)

    def preorder(self, node, result):
        if node:
            result.append(node.data.BuildingName)
            self.preorder(node.left, result)
            self.preorder(node.right, result)

    def postorder(self, node, result):
        if node:
            self.postorder(node.left, result)
            self.postorder(node.right, result)
            result.append(node.data.BuildingName)


#AVL TREE 

class AVLNode:
    def __init__(self, building):
        self.data = building
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    def height(self, node):
        return node.height if node else 0

    def balance(self, node):
        return self.height(node.left) - self.height(node.right) if node else 0

    def right_rotate(self, y):
        x = y.left
        t = x.right
        x.right = y
        y.left = t
        y.height = 1 + max(self.height(y.left), self.height(y.right))
        x.height = 1 + max(self.height(x.left), self.height(x.right))
        return x

    def left_rotate(self, x):
        y = x.right
        t = y.left
        y.left = x
        x.right = t
        x.height = 1 + max(self.height(x.left), self.height(x.right))
        y.height = 1 + max(self.height(y.left), self.height(y.right))
        return y

    def insert(self, node, building):
        if not node:
            return AVLNode(building)
        if building.BuildingID < node.data.BuildingID:
            node.left = self.insert(node.left, building)
        else:
            node.right = self.insert(node.right, building)

        node.height = 1 + max(self.height(node.left), self.height(node.right))
        bf = self.balance(node)

        if bf > 1 and building.BuildingID < node.left.data.BuildingID:
            return self.right_rotate(node)
        if bf < -1 and building.BuildingID > node.right.data.BuildingID:
            return self.left_rotate(node)
        if bf > 1 and building.BuildingID > node.left.data.BuildingID:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)
        if bf < -1 and building.BuildingID < node.right.data.BuildingID:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        return node


#GRAPH IMPLEMENTATION 

class Graph:
    def __init__(self, n):
        self.n = n
        self.matrix = [[0]*n for _ in range(n)]
        self.list = [[] for _ in range(n)]

    def add_edge(self, u, v, w):
        self.matrix[u][v] = w
        self.matrix[v][u] = w
        self.list[u].append((v, w))
        self.list[v].append((u, w))

    def bfs(self, start):
        visited = [False]*self.n
        queue = [start]
        visited[start] = True
        order = []
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v, w in self.list[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return order

    def dfs(self, start):
        visited = [False]*self.n
        stack = [start]
        order = []
        while stack:
            u = stack.pop()
            if not visited[u]:
                visited[u] = True
                order.append(u)
                for v, w in self.list[u]:
                    stack.append(v)
        return order

    def dijkstra(self, start):
        dist = [float('inf')]*self.n
        dist[start] = 0
        visited = [False]*self.n
        for _ in range(self.n):
            u = -1
            for i in range(self.n):
                if not visited[i] and (u == -1 or dist[i] < dist[u]):
                    u = i
            visited[u] = True
            for v, w in self.list[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
        return dist

    def kruskal(self):
        edges = []
        for u in range(self.n):
            for v, w in self.list[u]:
                if u < v:
                    edges.append((w, u, v))
        edges.sort()
        parent = list(range(self.n))

        def find(x):
            while x != parent[x]:
                x = parent[x]
            return x

        mst = []
        for w, u, v in edges:
            pu = find(u)
            pv = find(v)
            if pu != pv:
                mst.append((u, v, w))
                parent[pu] = pv
        return mst
    
    def visualize(self, title="Graph Visualization"):

        g = nx.Graph()
        for u in range(self.n):
            g.add_node(u)
            for v, w in self.list[u]:
                if u < v:
                    g.add_edge(u, v, weight=w)

        pos = nx.spring_layout(g)
        weights = nx.get_edge_attributes(g, 'weight')

        plt.figure(figsize=(6, 4))
        nx.draw(g, pos, with_labels=True, node_size=700, font_size=10)
        nx.draw_networkx_edge_labels(g, pos, edge_labels=weights)
        plt.title(title)
        plt.show()


#EXPRESSION TREE 

class ExpNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class ExpressionTree:
    def build(self, postfix):
        stack = []
        for ch in postfix:
            if ch.isdigit():
                stack.append(ExpNode(ch))
            else:
                node = ExpNode(ch)
                node.right = stack.pop()
                node.left = stack.pop()
                stack.append(node)
        return stack[-1]

    def evaluate(self, node):
        if node.val.isdigit():
            return int(node.val)
        l = self.evaluate(node.left)
        r = self.evaluate(node.right)
        if node.val == '+': return l + r
        if node.val == '-': return l - r
        if node.val == '*': return l * r
        if node.val == '/': return l // r


#CAMPUS NAVIGATION SYSTEM

class CampusSystem:
    def __init__(self):
        self.bst = BST()
        self.avl = AVLTree()
        self.bst_root = None
        self.avl_root = None
        self.graph = None
        self.exp_tree = ExpressionTree()

    def addBuildingRecord(self, id, name, location):
        b = Building(id, name, location)
        self.bst_root = self.bst.insert(self.bst_root, b)
        self.avl_root = self.avl.insert(self.avl_root, b)

    def listCampusLocations(self):
        inorder = []
        preorder = []
        postorder = []
        self.bst.inorder(self.bst_root, inorder)
        self.bst.preorder(self.bst_root, preorder)
        self.bst.postorder(self.bst_root, postorder)
        return inorder, preorder, postorder

    def constructCampusGraph(self, n, edges):
        self.graph = Graph(n)
        for u, v, w in edges:
            self.graph.add_edge(u, v, w)

    def findOptimalPath(self, start):
        return self.graph.dijkstra(start)

    def planUtilityLayout(self):
        return self.graph.kruskal()

    def evaluateExpression(self, postfix):
        root = self.exp_tree.build(postfix)
        return self.exp_tree.evaluate(root)


#test
if __name__ == "__main__":
    system = CampusSystem()
    system.addBuildingRecord(3, "Library", "Central Block")
    system.addBuildingRecord(1, "Admin", "Front Gate")
    system.addBuildingRecord(5, "Hostel", "Back Road")
    system.addBuildingRecord(2, "CSE Dept", "Tech Block")

    inorder, preorder, postorder = system.listCampusLocations()
    print("Inorder Traversal:", inorder)
    print("Preorder Traversal:", preorder)
    print("Postorder Traversal:", postorder)

    edges = [
        (0, 1, 4),
        (1, 2, 3),
        (2, 3, 2),
        (3, 4, 6),
        (4, 5, 5),
        (1, 4, 7),
        (0, 3, 9)
    ]
    system.constructCampusGraph(6, edges)

    print("BFS from Node 0:", system.graph.bfs(0))
    print("DFS from Node 0:", system.graph.dfs(0))
    print("Optimal Path Distances:", system.findOptimalPath(0))
    print("Planned Utility Layout (MST):", system.planUtilityLayout())


    expr = "23*54*+"
    print("Expression:", expr)
    print("Expression Result:", system.evaluateExpression(expr))

    system.graph.visualize("Campus Path Network")

