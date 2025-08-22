## 分布式训练

```
conda install -c conda-forge openmpi=4.0.5
conda install -c conda-forge mpi4py
```

#### 第 0 步：设置分布式文件系统

集群在机器之间同步文件夹

或者使用分布式文件系统，例如 NFS、Ceph

#### 第 1 步：设置 IP 配置文件

用户需要在训练前设置自己的 IP 配置文件 hostfile 。
例如，假设当前集群中有 2 台机器，则 IP 配置可能如下:
(示例文件已给出)
```
172.31.195.15 slots=1
172.31.195.16 slots=1
```

其中，slots 参数表示该机器启动的GPU数量

用户需要确保主节点具有正确的权限，可以无需密码验证即可 ssh 到所有其他节点。 

#### 第 2 步：对图进行分区

该示例提供了一个脚本，用于对一些图（例如 Cora 和 Reddit）进行分区。如果我们想在 2 台机器上训练 dist_gcn，则需要将图分区为 2 个部分。

```
python dist_partition2.py --dataset reddit --num_parts 2 --use_gdc
```

我们使用 Metis 在所有节点上将 reddit 乘积图划分为 2 个部分，该脚本生成分区图并将其存储在名为 data 的 reorder 的目录中

#### 第 3 步：启动分布式作业

```
mpirun -n 2 --hostfile hostfile \
--prefix /root/miniconda3/envs/jittor \
python dist_gcn.py --num_parts 2 --dataset reddit
```

mpirun: 是 MPI 标准的启动器命令

-n 2: 总共要启动 2 个进程

--hostfile hostfile: 指定一个主机文件

--prefix /root/miniconda3/envs/jittor: 指定 MPI 运行时环境的路径前缀，确保了每个节点上的进程都会在指定的 Conda 环境 (jittor) 中寻找依赖，从而保证了所有进程的运行环境一致

--num_parts 2：脚本的程序参数，这个数字必须和 -n 参数值保持一致


此外，我们可以在命令中再添加一些其他参数。

1. mpi 默认情况下不允许使用 root 进行训练，可以添加参数
--allow-run-as-root 来解决这个问题
```
mpirun -n 2 --hostfile hostfile \
--allow-run-as-root \
--prefix /root/miniconda3/envs/jittor \
python dist_gcn.py --num_parts 2 --dataset reddit
```

2. 显式启用 MPI 的 CUDA 支持
```
mpirun -n 2 --hostfile hostfile \
--prefix /root/miniconda3/envs/jittor \
--mca opal_cuda_support 1 \
python dist_gcn.py --num_parts 2 --dataset reddit
```

3. 显式指定网络接口（如 eth0 ）
```
mpirun -n 2 --hostfile hostfile \
--prefix /root/miniconda3/envs/jittor \
-x NCCL_SOCKET_IFNAME=eth0 \
python dist_gcn.py --num_parts 2 --dataset reddit
```

4. 需要在程序启动时打印出详细的 NCCL 初始化信息，用于DEBUG
```
mpirun -n 2 --hostfile hostfile \
--prefix /root/miniconda3/envs/jittor \
-x NCCL_DEBUG=INFO \
python dist_gcn.py --num_parts 2 --dataset reddit
```

#### 单机多卡训练

分图保存完毕后（如果有 conda 环境需要打开）
```
mpiexec -n 2 python dist_gcn.py --num_parts 2 --dataset reddit
```
直接使用这个命令即可开始单机多 GPU 训练

