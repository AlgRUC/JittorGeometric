<!--
 * @Description: 
 * @Author: lrl
 * @Date: 2024-07-01
-->

# Dataset_demo文档 (dataset_demo.md)
添加了dataset相关代码，目前包括节点分类以下数据集
* Plantoid
* Amazon
* Geom-GCN (Wikipedia, WebKB)
* ogb
此外，还包括动态图中链接预测任务的以下数据集
* JODIE (Wikipedia, Reddit, MOOC, LastFM)

# 目前的代码结构
Project
- jittor_geometric
- data
  - cora
  - ogbn_arxiv
  - ......

将数据集保存至/data后，使用以下代码进行测试：
```
python dataset_example.py --dataset $dataset_name 
```

# TODO
* 更多的数据集
* 代码整合
* xxx



