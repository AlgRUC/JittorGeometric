import pandas as pd
from jittor_geometric.data import Data
import os.path as osp
import numpy as np
from jittor_geometric.io.ogb_raw import read_csv_graph_raw, read_csv_heterograph_raw, read_binary_graph_raw, read_binary_heterograph_raw
from tqdm.auto import tqdm
import jittor as jt


def read_graph(raw_dir, add_inverse_edge=False, additional_node_files=[], additional_edge_files=[], binary=False):
    if binary:
        # npz
        graph_list = read_binary_graph_raw(raw_dir, add_inverse_edge)
    else:
        # csv
        graph_list = read_csv_graph_raw(raw_dir, add_inverse_edge, additional_node_files=additional_node_files, additional_edge_files=additional_edge_files)
    
    jittor_graph_list = []

    print('Converting graphs into Jittor objects...')

    for graph in tqdm(graph_list):
        g = Data()
        g.num_nodes = graph['num_nodes']
        g.edge_index = jt.array(graph['edge_index']).int()

        del graph['num_nodes']
        del graph['edge_index']

        if graph['edge_feat'] is not None:
            g.edge_attr = jt.array(graph['edge_feat'])
            del graph['edge_feat']

        if graph['node_feat'] is not None:
            g.x = jt.array(graph['node_feat'])
            del graph['node_feat']

        for key in additional_node_files:
            g[key] = jt.array(graph[key])
            del graph[key]

        for key in additional_edge_files:
            g[key] = jt.array(graph[key])
            del graph[key]

        jittor_graph_list.append(g)

    return jittor_graph_list

def read_heterograph(raw_dir, add_inverse_edge=False, additional_node_files=[], additional_edge_files=[], binary=False):
    if binary:
        # npz
        graph_list = read_binary_heterograph_raw(raw_dir, add_inverse_edge)
    else:
        # csv
        graph_list = read_csv_heterograph_raw(raw_dir, add_inverse_edge, additional_node_files=additional_node_files, additional_edge_files=additional_edge_files)

    jittor_graph_list = []

    print('Converting graphs into Jittor objects...')

    for graph in tqdm(graph_list):
        g = Data()
        
        g.__num_nodes__ = graph['num_nodes_dict']
        g.num_nodes_dict = graph['num_nodes_dict']

        # add edge connectivity
        g.edge_index_dict = {}
        for triplet, edge_index in graph['edge_index_dict'].items():
            g.edge_index_dict[triplet] = jt.array(edge_index).int()

        del graph['edge_index_dict']

        if graph['edge_feat_dict'] is not None:
            g.edge_attr_dict = {}
            for triplet in graph['edge_feat_dict'].keys():
                g.edge_attr_dict[triplet] = jt.array(graph['edge_feat_dict'][triplet])

            del graph['edge_feat_dict']

        if graph['node_feat_dict'] is not None:
            g.x_dict = {}
            for nodetype in graph['node_feat_dict'].keys():
                g.x_dict[nodetype] = jt.array(graph['node_feat_dict'][nodetype])

            del graph['node_feat_dict']

        for key in additional_node_files:
            g[key] = {}
            for nodetype in graph[key].keys():
                g[key][nodetype] = jt.array(graph[key][nodetype])

            del graph[key]

        for key in additional_edge_files:
            g[key] = {}
            for triplet in graph[key].keys():
                g[key][triplet] = jt.array(graph[key][triplet])

            del graph[key]

        jittor_graph_list.append(g)

    return jittor_graph_list

if __name__ == '__main__':
    pass