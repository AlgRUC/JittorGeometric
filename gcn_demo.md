<!--
 * @Description: 
 * @Author: lusz
 * @Date: 2024-06-22 19:45:13
-->
# GCN-demo文档 (gcn_demo.md)

## 改进点
- 使用C++算子进行图操作，加速计算
- 将图拓扑存储格式由COO改为CSC、CSR
- 调整反向传播为介于模型级和算子级之间的函数级，在函数中封装算子并定义`execute()`和`grad()`函数。

## 说明
- Jittor中使用元算子和不需要反向的算子无需定义`grad()`函数，但自定义的、需要反向的算子必须编写一个`grad()`函数。
- 在`data.py`中定义了CSC和CSR数据结构，可以通过`data.csc`和`data.csr`访问图拓扑。

## 待办事项
- 计算`edge_weight`的加速（仅适用于GCN算法）
- AVX向量化和多线程并行
- GPU加速、单机多卡、分布式计算
- `scatter_to_edge`和`gather_by_dst`用于支持GAT算法。

## 加速效果
xxx
