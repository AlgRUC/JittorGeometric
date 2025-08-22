'''
Description: Load and preprocess GNN datasets with relative paths.
'''
import os.path as osp
import argparse
import pickle
import os
from tqdm import tqdm
import sys
import jittor as jt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.data import DistChunk, CSR, CSC
from jittor_geometric.partition import DistChunkManager
from jittor_geometric.nn.conv.gcn_conv import gcn_norm
from jittor import Var
from jittor_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, OGBNodePropPredDataset, HeteroDataset, Reddit
import jittor_geometric.transforms as T
from pymetis import part_graph
import numpy as np

# 防止图太大超过内存限制
jt.flags.no_grad = 1 

def partition_graph(dataset_name, num_parts, use_gdc=False):
    """Partition a graph dataset using METIS."""
    script_dir = osp.dirname(osp.realpath(__file__))
    path = osp.join(script_dir, '..', '..','data')

    # Load dataset
    if dataset_name in ['computers', 'photo']:
        dataset = Amazon(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(path, dataset_name, geom_gcn_preprocess=False)
    elif dataset_name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
        dataset = OGBNodePropPredDataset(name=dataset_name, root=path)
    elif dataset_name in ['roman_empire', 'amazon_ratings', 'minesweeper', 'questions', 'tolokers']:
        dataset = HeteroDataset(path, dataset_name)
    elif dataset_name in ['reddit']:
        dataset = Reddit(os.path.join(path, 'Reddit'))
    
    data = dataset[0]
    edge_index = data.edge_index.numpy()
    num_nodes = data.x.shape[0]

    # METIS partition
    reorder_dir = osp.join(path, 'reorder', f"dist_{dataset_name}_{num_parts}part")

    chunk_manager = DistChunkManager(output_dir=reorder_dir)
    partition = chunk_manager.metis_partition(edge_index=edge_index, num_nodes=num_nodes, num_parts=num_parts)
    partition = np.array(partition)
    os.makedirs(reorder_dir, exist_ok=True)
    binary_file_path = osp.join(reorder_dir, f"dist_{dataset_name}_partition_{num_parts}.bin")
    if not osp.exists(binary_file_path):
        with open(binary_file_path, 'wb') as f:
            pickle.dump(partition, f)
        print("Partition file saved.")
    else:
        print("Partition file already exists. Skipping.")
    return dataset

def compute_chunk_id(src, chunk_offset):
    chunk_offset = chunk_offset.numpy() if isinstance(chunk_offset, jt.Var) else chunk_offset
    src = src.item() if isinstance(src, jt.Var) else src
    chunk_id = np.searchsorted(chunk_offset, src, side='right') - 1
    if chunk_id < 0 or chunk_id >= len(chunk_offset) - 1:
        chunk_id = len(chunk_offset) - 2
    return chunk_id

"""
将原图按照CSC进行压缩保存
这部分在gcn中没有用到，为时间考虑暂时注释掉，需要的时候再启动
"""
# def set_chunks_cscs(chunks, num_parts):
#     chunk_offset = chunks[0].offset
#     chunk_offset = chunk_offset.numpy() if isinstance(chunk_offset, jt.Var) else chunk_offset

#     for part in tqdm(range(num_parts), desc="Processing CSC partitions"):
#         current_chunk = chunks[part]
#         current_global_column_offset = current_chunk.CSC.column_offset.numpy() if isinstance(current_chunk.CSC.column_offset, jt.Var) else current_chunk.CSC.column_offset
#         current_global_row_indices = current_chunk.CSC.row_indices.numpy() if isinstance(current_chunk.CSC.row_indices, jt.Var) else current_chunk.CSC.row_indices
#         current_global_edge_weight = current_chunk.CSC.edge_weight.numpy() if isinstance(current_chunk.CSC.edge_weight, jt.Var) else current_chunk.CSC.edge_weight

#         chunk_column_offset_array = [[] for _ in range(num_parts)]
#         chunk_row_indices_array = [[] for _ in range(num_parts)]
#         chunk_edge_weight_array = [[] for _ in range(num_parts)]
#         index = [0] * num_parts
        
#         for i in range(num_parts):
#             chunk_column_offset_array[i].append(0)

#         for id in tqdm(range(current_chunk.v_num), desc=f"Processing nodes in partition {part}", leave=False):
#             for idx in range(current_global_column_offset[id], current_global_column_offset[id + 1]):
#                 src = current_global_row_indices[idx]
#                 chunk_id = np.searchsorted(chunk_offset, src, side='right') - 1
#                 if chunk_id < 0 or chunk_id >= num_parts:
#                     chunk_id = num_parts - 1
#                 src_adjusted = src - chunk_offset[chunk_id]
#                 edge_weight = current_global_edge_weight[idx]
#                 chunk_row_indices_array[chunk_id].append(src_adjusted)
#                 chunk_edge_weight_array[chunk_id].append(edge_weight)
#                 index[chunk_id] += 1

#             for i in range(num_parts):
#                 chunk_column_offset_array[i].append(chunk_column_offset_array[i][-1] + index[i])
#                 index[i] = 0

#         for i in range(num_parts):
#             row_indices = chunk_row_indices_array[i]
#             row_indices = jt.array(row_indices)
#             column_offset = chunk_column_offset_array[i]
#             column_offset = jt.array(column_offset)
#             edge_weight = chunk_edge_weight_array[i]
#             edge_weight = jt.array(edge_weight)
#             chunks[part].chunk_CSC.append(CSC(row_indices=row_indices, column_offset=column_offset, edge_weight=edge_weight))
        
#         import gc
#         gc.collect()
#         jt.gc()

def set_chunks_csrs(chunks, num_parts):
    chunk_offset = chunks[0].offset
    chunk_offset = chunk_offset.numpy() if isinstance(chunk_offset, jt.Var) else chunk_offset

    for part in tqdm(range(num_parts), desc="Processing CSR partitions"):
        current_chunk = chunks[part]
        current_global_row_offset = current_chunk.CSR.row_offset.numpy() if isinstance(current_chunk.CSR.row_offset, jt.Var) else current_chunk.CSR.row_offset
        current_global_column_indices = current_chunk.CSR.column_indices.numpy() if isinstance(current_chunk.CSR.column_indices, jt.Var) else current_chunk.CSR.column_indices
        current_global_edge_weight = current_chunk.CSR.edge_weight.numpy() if isinstance(current_chunk.CSR.edge_weight, jt.Var) else current_chunk.CSR.edge_weight

        chunk_row_offset_array = [[] for _ in range(num_parts)]
        chunk_column_indices_array = [[] for _ in range(num_parts)]
        chunk_edge_weight_array = [[] for _ in range(num_parts)]
        index1 = 0

        for i in range(num_parts):
            chunk_row_offset_array[i].append(0)

        for id in tqdm(range(current_chunk.global_v_num), desc=f"Processing nodes in partition {part}", leave=False):
            chunk_id = compute_chunk_id(id, chunk_offset)
            for idx in range(current_global_row_offset[id], current_global_row_offset[id + 1]):
                dst = current_global_column_indices[idx]
                weight = current_global_edge_weight[idx]
                chunk_column_indices_array[chunk_id].append(dst)
                chunk_edge_weight_array[chunk_id].append(weight)
                index1 += 1
            chunk_row_offset_array[chunk_id].append(chunk_row_offset_array[chunk_id][-1] + index1)
            index1 = 0

        for i in range(num_parts):
            column_indices = chunk_column_indices_array[i]
            column_indices = jt.array(column_indices)
            row_offset = chunk_row_offset_array[i]
            row_offset = jt.array(row_offset)
            edge_weight = chunk_edge_weight_array[i]
            edge_weight = jt.array(edge_weight)
            chunks[part].chunk_CSR.append(CSR(column_indices=column_indices, row_offset=row_offset, edge_weight=edge_weight))
        
        import gc
        gc.collect()
        jt.gc()

def save_to_chunk(dataset_name, dataset, num_parts, use_gdc=False):
    script_dir = osp.dirname(osp.realpath(__file__))
    path = osp.join(script_dir, '..', '..', 'data')

    # 数据集
    data = dataset[0]
    edge_index = data.edge_index.numpy()
    num_nodes = data.x.shape[0]
    num_classes = dataset.num_classes
    
    edge_weights = None
    edge_index, edge_weights = data.edge_index, data.edge_attr
    edge_index, edge_weights = gcn_norm(
                            edge_index, edge_weights ,num_nodes,
                            improved=False, add_self_loops=True)
    edge_index = edge_index.numpy()
    edge_weights = edge_weights.numpy()

    masks = []
    print("masks is creating")
    
    if dataset_name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        val_idx = split_idx['valid']
        test_idx = split_idx['test']

        train_mask = jt.zeros(num_nodes, dtype=jt.bool)
        train_mask[train_idx] = True

        val_mask = jt.zeros(num_nodes, dtype=jt.bool)
        val_mask[val_idx] = True

        test_mask = jt.zeros(num_nodes, dtype=jt.bool)
        test_mask[test_idx] = True

        masks.append(train_mask)
        masks.append(val_mask)
        masks.append(test_mask)
        
    else:
        masks.append(data.train_mask)
        masks.append(data.val_mask)
        masks.append(data.test_mask)

    # save to chunk files
    reorder_dir = osp.join(path, "reorder", f"dist_{dataset_name}_{num_parts}part_chunks")

    chunk_manager = DistChunkManager(reorder_dir)
    partition_file = osp.join(path, 'reorder',f'dist_{dataset_name}_{num_parts}part', f"partition_{num_parts}.bin")
    
    # result of first partition
    chunks = chunk_manager.partition_to_chunk(partition_file=partition_file, edge_index=edge_index, num_nodes=num_nodes, 
                                              num_parts=num_parts,edge_weight=edge_weights, 
                                              global_feature=data.x, global_label=data.y, global_masks=masks, 
                                              num_classes=num_classes)

    # second partition
    # set_chunks_cscs(chunks=chunks, num_parts=num_parts)
    set_chunks_csrs(chunks=chunks, num_parts=num_parts)
    
    # 保存结果到data
    chunk_manager.chunks_save(chunks, num_parts)

    print("\n--- Final Chunk Verification ---")
    for i in range(num_parts):
        print(f"Partition {i}: Node count (v_num) = {chunks[i].v_num}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gdc', action='store_true', help='Use GDC preprocessing.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the GNN dataset to load.')
    parser.add_argument('--num_parts', type=int, required=True, help='Partition number.')
    args = parser.parse_args()
    dataset = partition_graph(args.dataset, args.num_parts, args.use_gdc)
    save_to_chunk(args.dataset, dataset, args.num_parts, args.use_gdc)
