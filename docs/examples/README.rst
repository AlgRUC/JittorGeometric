运行示例 (Run Examples)
=======================

本项目提供了 **GCN（默认）**、**GraphSAGE（``--model sage``）** 和 **GAT（``--model gat``）** 三种模型，支持 **Cora**、**Reddit**、**OGBN-Arxiv** 数据集。

命令格式
--------

.. code-block:: bash

    python examples/minibatch_example.py         --dataset <DATASET_NAME>         --gpu         --batch_size <BATCH_SIZE>         --fanout <FANOUT_LIST>         --epoch <EPOCHS>         [--model {gcn|sage|gat}]

参数说明：

- ``--dataset`` : 数据集名称，可选值：``cora``、``reddit``、``ogbn-arxiv``
- ``--gpu`` : 使用 GPU 训练
- ``--batch_size`` : 每个 batch 的节点数量
- ``--fanout`` : 每层的邻居采样数（空格分隔表示多层）
- ``--epoch`` : 训练轮数（默认 50，部分数据集可选更大）
- ``--model`` : 模型类型，默认 ``gcn``，可选 ``sage`` 或 ``gat``

示例命令
--------

1. Cora 数据集
^^^^^^^^^^^^^^

.. code-block:: bash

    # GCN（默认）
    python examples/minibatch_example.py --dataset cora --gpu --batch_size 32 --fanout 3 5 --epoch 50

    # GraphSAGE
    python examples/minibatch_example.py --dataset cora --gpu --batch_size 32 --fanout 3 5 --epoch 50 --model sage

    # GAT
    python examples/minibatch_example.py --dataset cora --gpu --batch_size 32 --fanout 3 5 --epoch 50 --model gat

2. Reddit 数据集
^^^^^^^^^^^^^^^^

.. code-block:: bash

    # GCN（默认）
    python examples/minibatch_example.py --dataset reddit --gpu --fanout 10 10 --batch_size 512

3. OGBN-Arxiv 数据集
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # GCN（默认）
    python examples/minibatch_example.py --dataset ogbn-arxiv --gpu --fanout 10 10 --batch_size 512

    # GAT
    python examples/minibatch_example.py --dataset ogbn-arxiv --gpu --fanout 10 10 --batch_size 512 --model gat
