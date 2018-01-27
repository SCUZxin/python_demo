# -*-coding: utf-8 -*-

from collections import namedtuple
from sys import stdout

Node = namedtuple('Node', 'data, left, right')
tree = Node(1,
            Node(2,
                 Node(4,
                      Node(7, None, None),
                      None),
                 Node(5, None, None)),
            Node(3,
                 Node(6,
                      Node(8, None, None),
                      Node(9, None, None)),
                 None))

print(tree)

# 前序遍历
def preorder(node):
    if node is not None:
        print(node.data)
        preorder(node.left)
        preorder(node.right)

# 中序遍历
def inorder(node):
    if node is not None:
        inorder(node.left)
        print(node.data)
        inorder(node.right)

# 后序遍历
def postorder(node):
    if node is not None:
        postorder(node.left)
        postorder(node.right)
        print(node.data)

# 层序遍历
def levelorder(node, more=None):
    if node is not None:
        if more is None:
            more = []
        more += [node.left, node.right]
        print(node.data)
    levelorder(more[0], more[1:])


if __name__ == '__main__':
    print('preorder: '),
    preorder(tree)
    print('inorder: '),
    inorder(tree)
    print('postorder: '),
    postorder(tree)
    print('levelorder: '),
    levelorder(tree)
