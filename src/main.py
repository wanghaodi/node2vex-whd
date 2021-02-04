"""
Reference implementation of node2vec.

Author: Haodi Wang

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016
"""

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec


def parse_args():
    """
    Parses the node2vec arguments.
    """
    # 使用parser加载信息
    parser = argparse.ArgumentParser(description="Run node2vec.")

    # 输入文件
    parser.add_argument('--input', nargs='?', default='../graph/karate.edgelist',
                        help='Input graph path')

    # 输出文件
    parser.add_argument('--output', nargs='?', default='../emb/karate.emb',
                        help='Embeddings path')

    # embedding维度
    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    # 节点序列长度
    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    # 随机游走的次数
    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    # word2vex窗口大小，word2vex参数
    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    # SGD优化时的epoch数量，word2vex参数
    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    # 并行化核数，word2vex参数
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    # 参数p
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    # 参数q
    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    # 权重
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph():
    """
    Reads the input network in networkx.
    """
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


# 利用word2vex训练模型
def learn_embeddings(walks):
    """
    Learn embeddings by optimizing the Skipgram objective using SGD.
    """
    # 将node的类型int转为string
    # walks = [map(str, walk) for walk in walks] # 该行存在问题
    walk_lol = []
    for walk in walks:
        tmp = []
        for node in walk:
            tmp.append(str(node))
        walk_lol.append(tmp)
    # 调用gensim包运行word2vex
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)
    # 保存embedding信息
    model.wv.save_word2vec_format(args.output)

    return model


def main(args):
    """
    Pipeline for representational learning for all nodes in a graph.
    """
    nx_G = read_graph()
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks)


if __name__ == "__main__":
    args = parse_args()
    main(args)
