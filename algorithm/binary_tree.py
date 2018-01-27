# 问题: 二叉树查找

class Node:
    """
    二叉树左右枝，包括二叉树的初始化、插入节点、删除节点、寻找二叉树中某值、
    比较两棵树是否相同、打印二叉树的值等方法
    """
    def __init__(self, data):
        """
        节点结构
        :param data:根节点的value
        """
        self.left = None
        self.right = None
        self.data = data

    def insert(self, data):
        """
        插入节点数据
        :param data: 插入节点的value
        """
        if data < self.data:
            if self.left is None:
                self.left = Node(data)
            else:
                self.left.insert(data)
        elif data > self.data:
            if self.right is None:
                self.right = Node(data)
            else:
                self.right.insert(data)

    def lookup(self, data, parent=None):
        """
        遍历二叉树
        :param data:要寻找的节点的value
        :return: 返回data对应的节点和父节点
        """
        if data < self.data:
            if self.left is None:
                return None, None
            return self.left.lookup(data, self)
        elif data > self.data:
            if self.right is None:
                return None, None
            return self.right.lookup(data, self)
        else:
            return self, parent

    def delete(self, data):
        """
        删除节点
        :param data:要删除的节点的value
        :return: 返回删除节点后的二叉树
        """
        node, parent = self.lookup(data)
        # 要删除的节点存在
        if node is not None:
            children_count = node.children_count()
            # 如果该节点下没有子节点，即可删除
            if children_count == 0:
                if parent.left is node:
                    parent.left = None
                else:
                    parent.right = None
                del node
            # 如果该节点下有一个子节点，则让子节点代替该节点（该节点消失）
            elif children_count == 1:
                if node.left:
                    n = node.left   # n 暂存该子节点
                else:
                    n = node.right
                if parent:
                    if parent.left is node:
                        parent.left = n
                    else:
                        parent.right = n
            # 如果要删除的节点是root，则 parent is None
                else:
                    return n
                del node
            # 如果有两个子节点，要对子节点的数据进行判断，并重新安排节点排序
            else:
                parent = node
                successor = node.right
                while successor.left:
                    parent = successor
                    successor = successor.left
                node.data = successor.data
                if parent.left == successor:
                    parent.left = successor.right
                else:
                    parent.right = successor.right
        return node

    def compare_trees(self, node):
        """
        比较两棵树, 只要有一个节点（叶子）与另外一个树的不同，就返回False，也包括缺少对应叶子的情况。
        :param node: 与之比较的另外一棵树
        :return:True or False
        """
        if node is None:
            return False
        if self.data != node.data:
            return False
        res = True
        if self.left is None:
            if node.left:
                res = False
        else:
            res = self.left.compare_trees(node.left)
        if res is False:
            return False
        if self.right is None:
            if node.right:
                return False
        else:
            res = self.right.compare_trees(node.right)
        return res

    def tree_data(self):
        """
        二叉树数据结构
        """
        stack = []
        node = self
        while stack or node:
            if node:
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                yield node.data
                node = node.right

    def print_tree(self):
        """
        按顺序打印数的内容
        """
        if self.left:
            self.left.print_tree()
        print(self.data, end=" ")
        if self.right:
            self.right.print_tree()

    def children_count(self):
        """
        子节点个数
        """
        cnt = 0
        if self.left:
            cnt += 1
        if self.right:
            cnt += 1
        return cnt


root = Node(8)
root.insert(3)
root.insert(10)
root.insert(1)
root.insert(6)
root.insert(4)
root.insert(7)
root.insert(14)
root.insert(13)

root.print_tree()
print()
for data in root.tree_data():
    print(data)

print()
tree1 = Node(8)
tree1.insert(3)
tree1.insert(11)
tree2 = Node(8)
tree2.insert(3)
tree2.insert(11)
print(tree1.compare_trees(tree2))




