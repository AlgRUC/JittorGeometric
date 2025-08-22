
import sys,os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import os.path as osp
import argparse
import jittor as jt
import random 
import numpy as np
from jittor import nn


import math
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root)
from jittor_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, OGBNodePropPredDataset, HeteroDataset, Reddit
import jittor_geometric.transforms as T
from jittor_geometric.nn import GCNConv,GATConv,SAGEConv
import time
from jittor_geometric.ops import cootocsr,cootocsc
from jittor_geometric.nn.conv.gcn_conv import gcn_norm
from jittor_geometric.data import CSC, CSR
from jittor_geometric.ops import sampleprocessing,csctocsr,getweight,gpuinitco
from jittor_geometric.nn import GCNConv, SAGEConv
from jittor_geometric.nn.conv.sage_conv import sage_norm




jt.misc.set_global_seed(42)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='graph dataset')
parser.add_argument('--spmm', action='store_true', help='whether using spmm')
parser.add_argument('--gpu', action='store_true', help='use GPU (default: CPU)')
parser.add_argument('--batch_size', type=int, default=1024, help='mini-batch size')
parser.add_argument('--epoch', type=int, default=3, help='epoch')
parser.add_argument('--fanout', type=int, nargs='+', default=[25, 10],
                    help='neighbor sampling fanout for each layer')
parser.add_argument('--model', choices=['gcn', 'sage','gat'], default='gcn',
                    help='choose GNN model (gcn or gat)')
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()
dataset=args.dataset



jt.flags.use_cuda = 1 if args.gpu else 0

path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')


if dataset in ['computers', 'photo']:
    dataset = Amazon(path, dataset, transform=T.NormalizeFeatures())
elif dataset in ['cora', 'citeseer', 'pubmed']:
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
elif dataset in ['chameleon', 'squirrel']:
    dataset = WikipediaNetwork(path, dataset, geom_gcn_preprocess=True)
elif dataset in ['ogbn-arxiv','ogbn-products','ogbn-papers100M']:
    dataset = OGBNodePropPredDataset(name=dataset, root=path)
elif dataset in ['roman_empire', 'amazon_ratings', 'minesweeper', 'questions', 'tolokers']:
    dataset = HeteroDataset(path, dataset)
elif dataset in ['reddit']:
    dataset = Reddit(os.path.join(path, 'Reddit'))



data = dataset[0]
print(data)

v_num = data.x.shape[0]
edge_index, edge_weight = data.edge_index, data.edge_attr
if args.model == "gcn" or' args.model == "gat":':
    edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight,v_num,
                        improved=False, add_self_loops=True)
                    
elif args.model == "sage":
    edge_index, edge_weight = sage_norm(
                        edge_index, edge_weight,v_num,
                        improved=False, add_self_loops=True)

else:
    raise ValueError("Unsupported model type: {}".format(args.model))

with jt.no_grad():
    data.csc = cootocsc(edge_index, edge_weight, v_num)
    data.csr = cootocsr(edge_index, edge_weight, v_num)

if args.dataset in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
    split_idx = dataset.get_idx_split()
    train_nids = split_idx['train']
    valid_nids = split_idx['valid']
    test_nids  = split_idx['test']
    global_target = data.y.squeeze(1)


else:

    train_nids = jt.where(data.train_mask)[0]
    val_nids = jt.where(data.val_mask)[0]
    test_nids = jt.where(data.test_mask)[0]
    global_target = data.y



if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)

global_vertices = 0
global_edges   = 0

global_vertices = data.num_nodes  
global_edges = data.num_edges 


csc_global_column_offset = data.csc.column_offset
csc_global_row_indices = data.csc.row_indices
csr_global_row_offset = data.csr.row_offset
csr_global_column_indices = data.csr.column_indices

csc_global_edge_weight=data.csc.edge_weight
csr_global_edge_weight=data.csr.edge_weight


global_feature = data.x





class SampledSubgraph:
    # SampledSubgraph
    # 表示一个基于子图采样的图结构，包含多层采样子图及相关信息，
    def __init__(self):
        self.layer_graph = []  # 存储多个 LayerSampGraph，每个表示不同层的采样子图
        self.layers = 2 # 图的层数
        self.fanout =[10,10]
        #self.fanout =[3,5]
        self.X : jt.Var | None = None
        self.target_lab : jt.Var | None = None

        self.column_offset: jt.Var | None = None
        self.row_indices : jt.Var | None = None
        self.row_offset: jt.Var | None = None
        self.column_indices : jt.Var | None = None
        self.csc_edge_weight: jt.Var | None = None
        self.csr_edge_weight: jt.Var | None = None

        self.csc = None
        self.csr = None



        self.global2local = []  # 原图id -> 局部id Dict[int, int]
        self.local2global = []  # 局部id -> 原图id List[int]
        self.all_nodes = []
        self.num_nodes = 0  # 局部图的节点数
        self.num_edges = 0  # 局部图的边数

        self.edge_index: jt.Var | None = None  # 局部图的边索引
    
    def build_csc_csr(self):
        self.csc = CSC(
            row_indices=self.row_indices,
            column_offset=self.column_offset,
            edge_weight=self.csc_edge_weight 
        )
        self.csr = CSR(
            column_indices=self.column_indices,
            row_offset=self.row_offset,
            edge_weight=self.csr_edge_weight  
        )

        
      

class LayerSampGraph:
    # LayerSampGraph
    # 表示单层采样子图的结构，存储 CSC/CSR 所需的索引信息。
    def __init__(self):
        self.column_offset: jt.Var | None = None
        self.row_indices: jt.Var | None = None
        self.dst: jt.Var | None = None
        self.src: jt.Var | None = None



def build_full_ssg_from_data(data):
    ssg = SampledSubgraph()
    ssg.layers = 0   
    ssg.fanout = [-1, -1]  # 表示全邻居，实际上不是用来采样了

    # 加入特征和标签
    ssg.X=data.x  # 全图特征 [N, F]
    ssg.target_lab = data.y  # 全图标签 [N]

    ssg.csc = data.csc
    ssg.csr = data.csr


    return ssg


full_ssg = build_full_ssg_from_data(data)

       
def jt_shuffle(var: jt.Var) -> jt.Var:
    return var[jt.randperm(var.shape[0])]           


def csr_to_coo(row_off: jt.Var, col_idx: jt.Var):
    """
    CSR -> COO
    row_off: [num_nodes+1]
    col_idx: [num_edges]
    """
    num_nodes = row_off.shape[0] - 1
    counts = (row_off[1:] - row_off[:-1]).numpy()   # 出度
    src_np = np.repeat(np.arange(num_nodes), counts)  # 0..num_nodes-1 重复 counts
    dst_np = col_idx.numpy()

    edge_index_np = np.stack([src_np, dst_np], axis=0)
    return jt.array(edge_index_np, dtype=jt.int32)

def sample_one(ssg : SampledSubgraph,csc_global_column_offset:jt.Var, csc_global_row_indices:jt.Var,csr_global_row_offset:jt.Var, csr_global_column_indices:jt.Var,sample_nids:jt.Var):


    # 1采样子图。得到子图的csc和csr
    for i in range(ssg.layers):
        #1.1        
        csc_layer = LayerSampGraph()             

        #1.2 初始化csc_layer.dst。 如果是第一层，打乱；如果是其它层,等于上一层的采样结果(也是打乱)
        if i == 0:
            #开始已经打乱了
            csc_layer.dst=sample_nids
        else:
            csc_layer.dst = jt_shuffle(ssg.layer_graph[i - 1].src)


        

        #1.3 CPU 初始化这层的csc_layer.column_offset 
        dst = csc_layer.dst  # jt.Var, shape [N]
        fanout_i = ssg.fanout[i]
        nbrs = csc_global_column_offset[dst + 1] - csc_global_column_offset[dst]
        # 限制采样数
        if fanout_i < 0:
            sampled_nbrs = nbrs
        else:
            sampled_nbrs = jt.minimum(nbrs, jt.full_like(nbrs, fanout_i))
        # 构建 column_offset
        cumsum_offsets = jt.cumsum(sampled_nbrs, dim=0)
        csc_layer.column_offset = jt.concat([jt.array([0], dtype=jt.int32), cumsum_offsets], dim=0)



        
        #1.4 cpu gpu采样处理过程,填充csc_layer的row_indices 采样处理耗时: 0.0142秒 采样处理耗时: 0.0018秒
        jt.flags.lazy_execution = 0  # 关闭懒执行
        csc_layer.row_indices=sampleprocessing(ssg.fanout[i], csc_layer.dst, csc_layer.column_offset,csc_global_column_offset,csc_global_row_indices)


     
        #1.5 CPU初始化csc_layer.src。这个src是源节点id   

        #方法2

        csc_layer.src = jt.unique(csc_layer.row_indices)  # 去重。还有排序的效果
        ssg.all_nodes.append(csc_layer.dst)  # 保存当前层的dst节点
        ssg.all_nodes.append(csc_layer.src)  # 保存当前层的src节点


        ssg.layer_graph.append(csc_layer)


    #2 得到小图所有节点 

    all_nodes_concat = jt.concat(ssg.all_nodes, dim=0)
    global_nodes = jt.unique(all_nodes_concat)
    ssg.local2global = global_nodes  # 局部id -> 原图id List[int]



    max_id = global_nodes.max().item() 
    global2local = -1 * jt.ones((max_id+ 1,), dtype=jt.int32)  # 默认填充 -1
    global2local[global_nodes] = jt.arange(global_nodes.shape[0], dtype=jt.int32)
    ssg.global2local = global2local





    #3 构造小图csc


    # 3.1. 初始化邻接表（局部id表示）0.05-0.07

    from collections import defaultdict    
    adj_dict = defaultdict(list)  # Dict[int, List[int]]

    for layer in ssg.layer_graph:
        col_offset = layer.column_offset.numpy()
        row_indices = layer.row_indices.numpy()
        dst_nodes = layer.dst.numpy()
        global2local_np = ssg.global2local.numpy()  # 一次性转换为局部 ID

        dst_locals = global2local_np[dst_nodes]  # 一次性转换

        for i in range(len(dst_nodes)):
            dst_local = dst_locals[i]
            start = col_offset[i]
            end = col_offset[i + 1]

            src_globals = row_indices[start:end]  # 一次取出多个邻居
            src_locals =global2local_np[src_globals] # 一次性转换为局部 ID

            adj_dict[dst_local].extend(src_locals.tolist())  # 添加到邻接表中




    
    # 3.2. 按照 dst_local 排序生成最终的 CSC 格式 0.06


    ssg.num_nodes = len(ssg.local2global)
    column_offsets = [0]*(ssg.num_nodes + 1)  # 初始化列偏移量
    row_indices = []
    for dst_local in range(ssg.num_nodes):  # 顺序必须是局部 id 排序
        neighbors = adj_dict.get(dst_local, [])
        row_indices.extend(neighbors)
        #column_offsets.append(len(row_indices))  # 当前累积边数
        column_offsets[dst_local + 1] = column_offsets[dst_local]+ len(neighbors) # 更新列偏移量

    ssg.column_offset = jt.array(column_offsets)
    ssg.row_indices =jt.array( row_indices)




    #4  cpu gpu 构造小图csr
   
    ssg.row_offset,ssg.column_indices= csctocsr(ssg.column_offset,ssg.row_indices,jt.arange(ssg.num_nodes),jt.arange(ssg.num_nodes))



    #5. cpu gpu更新权重 

    ssg.csc_edge_weight=getweight(ssg.num_nodes,jt.arange(ssg.num_nodes),jt.arange(ssg.num_nodes),ssg.column_offset,ssg.row_indices)
    ssg.csr_edge_weight=getweight(ssg.num_nodes,jt.arange(ssg.num_nodes),jt.arange(ssg.num_nodes),ssg.row_offset,ssg.column_indices)
    

 
    #6 保存ssg的X[0],也就是feature 
   
    ssg.X= global_feature[ssg.local2global]  # 根据 local2global 映射提取特征

  

    #7 保存ssg的target_lab   

    ssg.target_lab = global_target[ssg.local2global]  # 使用 global_target 提取对应节点的标签

    ssg.build_csc_csr()
 
    

    return ssg  # 返回采样的子图对象
    
class NeighborLoader:
    def __init__(self, nids, fanout, batch_size,
                 csc_col_off, csc_row_idx, csr_row_off, csr_col_idx):
        self.nids = nids
        self.fanout = fanout
        self.batch_size = batch_size
        self.csc_col_off = csc_col_off
        self.csc_row_idx = csc_row_idx
        self.csr_row_off = csr_row_off
        self.csr_col_idx = csr_col_idx
        self.batch_ids = []
        #self.reset()

    def reset(self):
        self.nids = jt_shuffle(self.nids)  # 打乱 jt.Var 类型的 nids
        self.batch_ids = [
            self.nids[i : i + self.batch_size]
            for i in range(0, self.nids.shape[0], self.batch_size)
        ]


    def __len__(self):
        return len(self.batch_ids)
        

    def __iter__(self):
        for batch_nids in self.batch_ids:
            ssg = SampledSubgraph()
            ssg.fanout = self.fanout
            ssg = sample_one(ssg,
                             self.csc_col_off,
                             self.csc_row_idx,
                             self.csr_row_off,
                             self.csr_col_idx,
                             batch_nids)
            yield ssg


train_loader = NeighborLoader(
    nids=train_nids,
    fanout=args.fanout,
    batch_size=args.batch_size,
    csc_col_off=csc_global_column_offset,
    csc_row_idx=csc_global_row_indices,
    csr_row_off=csr_global_row_offset,
    csr_col_idx=csr_global_column_indices,
)



class GCN(nn.Module):
    def __init__(self, dataset, dropout=0.8):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels=dataset.num_features, out_channels=256,spmm=args.spmm)
        self.conv2 = GCNConv(in_channels=256, out_channels=dataset.num_classes,spmm=args.spmm)
        self.dropout = dropout

    
    def execute(self, ssg):
        #jt.flags.use_cuda = 0  # 禁用 GPU
        x = ssg.X # [2708,1433,]     
        csc = ssg.csc
        csr = ssg.csr


        x = nn.relu(self.conv1(x, csc, csr))
        x = nn.dropout(x, self.dropout, is_train=self.training)
        x = self.conv2(x, csc, csr)
        return nn.log_softmax(x, dim=1)

class SAGE(nn.Module):
    def __init__(self, dataset, dropout=0.8):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels=dataset.num_features, out_channels=256, cached = True, root_weight = False, spmm=args.spmm)
        self.conv2 = SAGEConv(in_channels=256, out_channels=dataset.num_classes, cached = True, root_weight = False, spmm=args.spmm)
        self.dropout = dropout

    def execute(self, ssg):
        
        x = ssg.X    
        csc = ssg.csc
        csr = ssg.csr


        x = nn.relu(self.conv1(x, csc=csc, csr=csr))
        x = nn.dropout(x, self.dropout, is_train=self.training)
        x = self.conv2(x, csc=csc, csr=csr)
        return nn.log_softmax(x, dim=1)
    
class GAT(nn.Module):
    def __init__(self, dataset):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dataset.num_features,128,global_edges, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GATConv(128, dataset.num_classes,global_edges, cached=True,
                             normalize=not args.use_gdc)

    def execute(self, ssg):
        x = ssg.X    
        csc = ssg.csc
        x = nn.relu(self.conv1(x, csc))
        x = nn.dropout(x)
        x = nn.relu(self.conv2(x,csc))
        return nn.log_softmax(x, dim=1)



if args.model == "gcn":
    model = GCN(dataset)
    optimizer = nn.Adam(params=model.parameters(), lr=0.001, weight_decay=5e-4)
elif args.model == "sage":
    model =SAGE(dataset)
    optimizer = nn.Adam(params=model.parameters(), lr=0.001, weight_decay=5e-4) 
elif args.model == "gat":
    model = GAT(dataset)
    optimizer = nn.Adam([
    dict(params=model.conv1.parameters(), weight_decay=1e-4),
    dict(params=model.conv2.parameters(), weight_decay=1e-4)
], lr=5e-3)
else:
    raise ValueError("Unsupported model type: {}".format(args.model))




def train(my_loader):
    model.train()
    total_loss = 0.0
    total_sampling_time = 0.0
    total_training_time = 0.0

    loader_iter = iter(my_loader)

    for _ in range(len(my_loader)):
        # -------- 采样计时 --------
        sample_start = time.time()
        ssg = next(loader_iter)
        sample_end = time.time()
        sample_time = sample_end - sample_start
        total_sampling_time += sample_time



        # -------- 训练计时 --------
        train_start = time.time()
        optimizer.zero_grad()
        pred = model(ssg)
        label = ssg.target_lab

        loss = nn.nll_loss(pred, label)
        total_loss += loss.item()
        optimizer.step(loss)
        train_end = time.time()
        total_training_time += (train_end - train_start)





    return total_loss / len(my_loader),total_sampling_time, total_training_time





def test():
    model.eval()
    logits = model(full_ssg)  # logits 是 jittor.Var，shape 可能是 (N, C)
    pred_label = jt.argmax(logits, dim=1) # 预测标签转numpy
    pred_label = pred_label[0]  # 假设full_ssg只有一个图，取第一个图的预测标签
    true_label = full_ssg.target_lab  # 

    accs = []
    if args.dataset in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
        true_label = true_label.squeeze()  # 
        for idx in [train_nids, valid_nids, test_nids]:
            acc = (pred_label[idx] == true_label[idx]).float32().mean()
            accs.append(acc)
    else:
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            mask_np = mask.astype(bool)
            acc = (pred_label[mask_np] == true_label[mask_np]).sum() / mask_np.sum()
            accs.append(acc)

    return accs





## --- 训练过程 ---
best_val_acc = test_acc = 0
num_epochs = args.epoch
total_sampling_time = 0.0
total_training_time = 0.0

start = time.time()

for epoch in range(1, num_epochs + 1):

    epoch_start = time.time()
    resettime_start = time.time()
    train_loader.reset()
    resettime_end = time.time()
    #print(f"Epoch {epoch} reset time: {resettime_end - resettime_start:.4f} seconds")
    # --- 统计训练时间 ---
    
    traintime_start= time.time()
    epoch_loss,epoch_sampling_time, epoch_training_time =train(train_loader)
    traintime_end = time.time()
    #print(f"Epoch {epoch} train 函数 time: {traintime_end - traintime_start:.4f} seconds")
    #print(f"Epoch {epoch} Loss: {epoch_loss:.4f}")
    #print(f"Epoch {epoch} Sampling Time: {epoch_sampling_time:.4f} seconds")
    #print(f"Epoch {epoch} Training Time: {epoch_training_time:.4f} seconds")

    total_sampling_time += epoch_sampling_time
    total_training_time += epoch_training_time

    jt.sync_all()  # 
    
    

    # --- 测试 ---
    testtime_start = time.time()
    train_acc, val_acc, tmp_test_acc = test()
    testtime_end = time.time()
    #print(f"Epoch {epoch} Test Time: {testtime_end - testtime_start:.4f} seconds")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    epoch_end = time.time()
    epoch_time = (epoch_end - epoch_start)
    #print(f"Epoch {epoch} took {epoch_time:.4f} seconds")
    print(log.format(epoch, train_acc, best_val_acc, test_acc))

jt.sync_all()
end = time.time()

# --- 平均时间 ---
avg_sampling_time = total_sampling_time / num_epochs
avg_training_time = total_training_time / num_epochs
avg_time= (end - start) / num_epochs


print(f"Average sampling time per epoch: {avg_sampling_time:.4f} seconds")
print(f"Average training time per epoch: {avg_training_time:.4f} seconds")
print(f"Average time per epoch: {avg_time:.4f} seconds")
#print(f"Total training+sampling time: {end - start:.4f} seconds")
    

    



   


    




  
                
    
    
            





        




    



         

    



   


    




  
                
    
    
            





        




    




         