# 动态图文档
## 添加文件汇总
#### 动态图的主要部分都写在`dygstore`文件夹下：
* `adjlist.py`:主要包含动态图的存储结构
* `embedding.py`:一个基于单例模式的动态节点嵌入管理系统
* `dysubgraph.py`: 每个事件所对应的动态子图
* `dependedgraph.py`: 根据一批动态子图生成依赖分析图，返回可以并行执行的事件批次
* `parallel.py`: 并行执行模块，将需要并行执行的部分封装成函数发给其即可
#### 其他改动：
* `dyrepbase.py`: 基础模型 在models文件夹下
* `dyrepupdate.py`: 并行模型 在models文件夹下
* `dyrep_example2.py`: 基础模型的示例 在examples文件夹下
* `dyrep_example3.py`: 并行模型的示例 在examples文件夹下
* 在`datasets`文件夹中，新增了一个`github`数据集，使用时通过`github.py`。
以上是对现有动态图工作的简要介绍，接下来将依次展开对每个部分的详细介绍

## adjlist.py

`adjlist.py` 文件主要实现了动态图的存储结构，提供了有向图的邻接表表示及其相关操作。以下是主要的类、方法和功能介绍：

---

### 1. `EdgeData`
- **描述**：表示边的属性数据。
- **属性**：
  - `value`：边的权重，默认为 `1.0`。
- **方法**：
  - `to_string()`：将边的数据转换为字符串。
- **示例**：
  ```python
  edge_data = EdgeData(value=2.5)
  print(edge_data.to_string())  # 输出: "2.5"

### 2. `TEdge`
- 表示一条有向边，包含以下属性：
  - `src`: 边的起点。
  - `dst`: 边的终点。
  - `edge_data`: 边的属性数据，默认为 `EdgeData`。
  - `time`: 边的时间戳，默认为当前时间。
- 提供了 `__repr__` 方法，用于打印边的信息。

### 3. `DanceTEdge`
- 表示一个节点的邻居集合，支持高效的邻居管理。
- 主要功能：
  - `add_neighbor` 和 `add_neighbors`: 添加单个或多个邻居。
  - `is_edge`: 检查某个节点是否是邻居。
  - `get_ids` 和 `get_edge_datas`: 获取所有邻居的 ID 和对应的边数据。
  - `size`: 返回邻居的数量。

### 4. `AdjTableNodes`
- 表示一个节点的入边和出边集合。
- 主要功能：
  - `add_in_edge` 和 `add_out_edge`: 添加单个入边或出边。
  - `add_in_edges` 和 `add_out_edges`: 添加多个入边或出边。
  - `has_in_edge` 和 `has_out_edge`: 检查是否存在某个入边或出边。
  - `in_neighbors_ids` 和 `out_neighbors_ids`: 获取入邻居和出邻居的 ID。
  - `num_in_neighbors` 和 `num_out_neighbors`: 获取入邻居和出邻居的数量。

### 5. `DiAdjList`
- 表示整个有向图的邻接表。
- 主要功能：
  - **顶点管理**：
    - `add_vertexes`: 添加多个顶点。
  - **边管理**：
    - `add_edge`: 添加一条边。
    - `create_base_edges`: 批量添加边，支持并行处理。
    - `out_edges` 和 `in_edges`: 获取某个节点的出边和入边。
    - `has_edge`: 检查是否存在某条边。
  - **邻居管理**：
    - `out_neighbors` 和 `in_neighbors`: 获取某个节点的出邻居和入邻居。
  - **度计算**：
    - `degree`: 获取某个节点的总度数。
    - `in_degree` 和 `out_degree`: 获取某个节点的入度和出度。
  - **清理操作**：
    - `clear`: 清空图的所有数据。
  - **并行处理**：
    - 使用 `ThreadPoolExecutor` 并行处理边的添加操作，提高效率。

### 使用示例
以下是一个简单的使用示例：
```python
from adjlist import DiAdjList, TEdge

# 创建一个有向图
graph = DiAdjList()

# 添加顶点
graph.add_vertexes(5)

# 添加边
edge1 = TEdge(src=0, dst=1)
edge2 = TEdge(src=1, dst=2)
graph.add_edge(edge1)
graph.add_edge(edge2)

# 获取节点的出边和入边
print("Node 0 out edges:", graph.out_edges(0))
print("Node 1 in edges:", graph.in_edges(1))

# 检查边是否存在
print("Edge (0 -> 1) exists:", graph.has_edge(0, 1))
print("Edge (1 -> 3) exists:", graph.has_edge(1, 3))

# 获取节点的度
print("Node 1 degree:", graph.degree(1))
```
## embedding.py

`embedding.py` 文件实现了一个基于单例模式的动态节点嵌入管理系统，支持嵌入的初始化、更新、查询、合并以及重置操作。以下是主要的类和功能介绍：

---

### 1. `InitialEmb`
- **描述**：用于存储初始的节点嵌入，支持单例模式。
- **属性**：
  - `initial_embs`：初始嵌入列表。
  - `num_nodes`：节点数量。
  - `hidden_dim`：嵌入维度。
  - `device`：设备信息。
- **方法**：
  - `get_instance()`：获取单例对象。
  - `initialize(num_nodes, hidden_dim, device, initial_embs)`：初始化嵌入。
  - `clear()`：清空初始嵌入。
- **示例**：
  ```python
  initial_emb = InitialEmb.get_instance()
  initial_emb.initialize(num_nodes=100, hidden_dim=64, device="cpu", initial_embs=[jt.randn(100, 64)])
  print(initial_emb.num_nodes)  # 输出: 100
  ```

### 2. `DyNodeEmbedding`
- **描述**：动态节点嵌入管理器，支持嵌入的动态更新和查询，采用单例模式。
- **属性**：
  - `m_embedding`：当前嵌入列表，每个维度的嵌入存储为一个 `jittor.Var`。
  - `m_update_count`：记录每个节点的更新次数，使用 `defaultdict` 存储。
  - `m_update_emb`：存储每次更新的嵌入，按维度和版本号组织。
  - `is_merge`：标记是否需要合并更新。
  - `m_num_nodes`：节点数量。
  - `m_hidden_dim`：嵌入维度。
  - `m_device`：设备信息。
  - `rw_lock`：读写锁，确保多线程环境下的安全性。
- **方法**：
  - `get_instance()`：获取单例对象。
  - `inits(embs)`：初始化嵌入。
  - `index(uid, dim=0)`：查询某个节点的嵌入。
  - `update(uid, embedding, dim=0)`：更新某个节点的嵌入。
  - `merge()`：合并所有更新到主嵌入。
  - `back_to_initial()`：重置嵌入到初始状态。
  - `_get_version_id(dim, index)`：获取某个节点的更新版本号。
  - `_index_update(vid, nid, dim_id)`：查询某个节点的某次更新嵌入。
  - `_update_emb(vid, nid, dim_id, emb)`：记录某次更新的嵌入。
  - `_clear_updates()`：清空所有更新记录。

---

### 使用示例
以下是一个完整的使用示例，展示如何使用 `DyNodeEmbedding` 类管理动态节点嵌入：

```python
from embedding import DyNodeEmbedding
import jittor as jt

# 初始化嵌入
initial_embs = [jt.randn(100, 64) for _ in range(3)]  # 三个维度的嵌入
emb_manager = DyNodeEmbedding.get_instance()
emb_manager.inits(initial_embs)

# 更新嵌入
new_emb = jt.randn(64)  # 新的嵌入向量
emb_manager.update(10, new_emb, dim=0)  # 更新节点 10 的第 0 维嵌入

# 查询嵌入
result = emb_manager.index(10, dim=0)  # 查询节点 10 的第 0 维嵌入
print("Embedding shape:", result.shape)

# 合并更新
emb_manager.merge()  # 将所有更新合并到主嵌入

# 重置到初始状态
emb_manager.back_to_initial()  # 将嵌入重置为初始状态
```

## dysubgraph.py

`dysubgraph.py` 文件实现了动态子图的构建和管理。动态子图是基于事件的局部图结构，支持邻居扩展、时间戳管理以及动态更新。以下是主要的类和功能介绍：

---

### 1. `DySubGraph`
- **描述**：动态子图构造器，用于根据事件生成局部子图。
- **属性**：
  - `u`：源节点 ID。
  - `v`：目标节点 ID。
  - `event_type`：事件类型（用于 `EdgeData`）。
  - `timestamp`：事件时间戳。
  - `di_adjlist`：全局邻接表引用（`DiAdjList` 对象）。
  - `hop`：邻居扩展跳数。
  - `node_set`：子图包含的节点集合。
  - `time_map`：节点的最后活跃时间映射。
- **方法**：
  - `_add_base_edges()`：添加事件的基础边到全局邻接表。
  - `_expand_subgraph()`：递归扩展子图的邻居节点。
  - `_get_neighbors(node)`：获取某个节点的出邻居和入邻居。
  - `nodes`：返回子图的所有节点。
  - `update_nodes`：返回事件中涉及的节点（`u` 和 `v`）。
  - `get_neighbors(node)`：获取某个节点的所有邻居（出邻居和入邻居）。
  - `get_last_active(node)`：获取某个节点的最后活跃时间。
  - `event`：返回事件的基本信息（源节点、目标节点、事件类型、时间戳）。
  - `update_timestamp(node, new_time)`：更新节点的时间戳。
  - `edge_count`：计算子图的边数。
  - `density()`：计算子图的密度。

---

### 使用示例
以下是一个完整的使用示例，展示如何使用 `DySubGraph` 类构建和管理动态子图：

```python
from dysubgraph import DySubGraph
from adjlist import DiAdjList
from datetime import datetime

# 初始化全局邻接表
global_adj = DiAdjList()
global_adj.add_vertexes(1000)  # 预分配节点空间

# 创建动态子图
event_subg = DySubGraph(
    u=123,
    v=456,
    event_type=1,
    timestamp=datetime.now(),
    di_adjlist=global_adj,
    hop=2  # 扩展两跳邻居
)

# 查询子图信息
print("包含节点:", event_subg.nodes)
print("节点123的邻居:", event_subg.get_neighbors(123))
print("最后活跃时间:", event_subg.get_last_active(456))

# 更新节点时间戳
event_subg.update_timestamp(123, datetime.now())

# 验证全局影响
print("全局邻接表中是否存在边(123,456):", 
      global_adj.has_edge(123, 456))  # 应返回True

# 计算子图密度
print("子图密度:", event_subg.density())
```

## dependedgraph.py

`dependedgraph.py` 文件实现了依赖关系图的构建和管理。依赖关系图用于分析事件之间的依赖关系，并生成可以并行执行的事件批次。以下是主要的类和功能介绍：

---

### 1. `DependedGraph`
- **描述**：依赖关系图分析系统，用于分析事件之间的依赖关系并生成执行层次。
- **属性**：
  - `graph`：存储依赖关系的图结构，格式为 `{事件ID: {'deps': 依赖的事件集合, 'deped': 被依赖的事件集合}}`。
  - `start_id`：起始事件 ID，通常为 `0`。
  - `total_events`：总事件数量，用于预分配内存。
  - `execution_layers`：存储事件的执行层次，每一层包含可以并行执行的事件 ID。
  - `current_layer`：当前执行层次的索引。
  - `lock`：线程锁，确保多线程环境下的安全性。
  - `dependency_cache`：缓存依赖关系的检测结果，用于优化性能。
- **方法**：
  - `_init_super_node()`：初始化虚拟超级节点，用于统一管理所有事件的依赖关系。
  - `add_dependency(src, dst)`：添加两个事件之间的依赖关系。
  - `analyze_batch(subgraphs, max_workers=10)`：批量分析事件之间的依赖关系。
  - `_analyze_single(prev, current, check_id, check_id1)`：分析两个事件之间的依赖关系。
  - `_build_execution_layers()`：基于拓扑排序构建事件的执行层次。
  - `get_parallel_batch()`：获取下一批可以并行执行的事件 ID。
  - `visualize()`：可视化依赖关系图和执行层次。

---

### 使用示例
以下是一个完整的使用示例，展示如何使用 `DependedGraph` 类分析事件依赖关系并生成执行批次：

```python
from dependedgraph import DependedGraph
from dysubgraph import DySubGraph
from adjlist import DiAdjList
from datetime import datetime, timedelta

# 初始化全局邻接表
global_adj = DiAdjList()
global_adj.add_vertexes(1000)  # 预分配节点空间

# 构造一组动态子图，模拟不同的事件
base_time = datetime.now()
subgraphs = [
    DySubGraph(u=0, v=1, event_type=1, timestamp=base_time, di_adjlist=global_adj, hop=1),
    DySubGraph(u=1, v=2, event_type=1, timestamp=base_time + timedelta(seconds=1), di_adjlist=global_adj, hop=1),
    DySubGraph(u=2, v=3, event_type=1, timestamp=base_time + timedelta(seconds=2), di_adjlist=global_adj, hop=1),
    DySubGraph(u=4, v=5, event_type=1, timestamp=base_time + timedelta(seconds=3), di_adjlist=global_adj, hop=1),
    DySubGraph(u=3, v=1, event_type=1, timestamp=base_time + timedelta(seconds=4), di_adjlist=global_adj, hop=1),
]

# 创建依赖关系图
dep_graph = DependedGraph(total_events=len(subgraphs), start_id=0)

# 批量分析依赖关系
dep_graph.analyze_batch(subgraphs, max_workers=4)

# 可视化依赖关系图
dep_graph.visualize()

# 获取并行执行批次
print("\n===== Parallel Execution Batches =====")
batch = dep_graph.get_parallel_batch()
batch_num = 0
while batch is not None:
    print(f"Batch {batch_num}: {sorted(batch)}")
    batch = dep_graph.get_parallel_batch()
    batch_num += 1
```

## parallel.py

`parallel.py` 文件实现了并行执行模块，支持事件级别的并行处理和分析-处理流水线。以下是主要的类和功能介绍：

---

### 1. `ParallelProcessor`
- **描述**：并行处理器，用于管理分析和处理任务的并行执行。
- **属性**：
  - `max_analyzer_threads`：分析任务的最大线程数。
  - `max_processor_threads`：处理任务的最大线程数。
  - `max_event_workers`：事件级并行处理的最大工作线程数。
  - `queue_maxsize`：分析器和处理器之间队列的最大容量。
  - `use_processes`：是否使用 `ProcessPoolExecutor`（默认为 `False`，使用 `ThreadPoolExecutor`）。
  - `analyzer_to_processor_queue`：分析器和处理器之间的队列。
  - `process_results_queue`：存储处理结果的队列。
- **方法**：
  - `_run_analyzer(analyzer_func, chunks, dyrep_obj)`：运行分析器函数，将结果推送到队列。
  - `_run_processor(processor_func, dyrep_obj, chunks)`：运行处理器函数，将结果推送到结果队列。
  - `execute_pipeline(analyzer_func, processor_func, dyrep_obj, chunks)`：执行分析器和处理器的流水线。
  - `process_events_parallel(process_event_func, batch_size, shared_lock=None)`：使用线程池或进程池并行处理事件。
  - `_wrap_event_func(process_event_func, idx, shared_lock)`：包装事件处理函数，支持锁机制。

---

### 使用示例
以下是一个完整的使用示例，展示如何使用 `ParallelProcessor` 类实现分析和处理任务的并行执行：

```python
from parallel import ParallelProcessor
from queue import Queue
import jittor as jt

# 示例分析器函数
def analyzer_func(chunks, dyrep_obj, analyze_to_process_queue):
    for chunk in chunks:
        # 模拟分析任务
        analyze_to_process_queue.put(chunk)
    analyze_to_process_queue.put(None)  # 结束信号

# 示例处理器函数
def processor_func(dyrep_obj, analyze_to_process_queue, process_results_queue, chunks):
    results = []
    while True:
        item = analyze_to_process_queue.get()
        if item is None:
            break
        # 模拟处理任务
        results.append(item)
    process_results_queue.put(results)

# 示例事件处理函数
def process_event_func(idx):
    return f"Processed event {idx}"

# 初始化并行处理器
parallel_processor = ParallelProcessor(
    max_analyzer_threads=2,
    max_processor_threads=2,
    max_event_workers=4,
    queue_maxsize=10,
    use_processes=False
)

# 示例数据
chunks = [list(range(10)), list(range(10, 20))]  # 模拟数据块
dyrep_obj = None  # 示例中不需要具体的 DyRep 对象

# 执行分析和处理流水线
results = parallel_processor.execute_pipeline(
    analyzer_func=analyzer_func,
    processor_func=processor_func,
    dyrep_obj=dyrep_obj,
    chunks=chunks
)
print("Pipeline results:", results)

# 并行处理事件
batch_size = 10
event_results = parallel_processor.process_events_parallel(
    process_event_func=process_event_func,
    batch_size=batch_size
)
print("Event results:", event_results)

```

## 示例
在 `examples` 文件夹中，提供了一个完整的动态图示例，展示了如何使用上述模块构建和管理动态图。其中 `dyrep_example2.py` 是一个普通版本的 DyRep 模型（根据论文 "DYREP: LEARNING REPRESENTATIONS OVER DYNAMIC GRAPHS" 和参考 https://github.com/Harryi0/dyrep_torch 实现），而 `dyrep_example3.py` 是一个基于并行处理的 DyRep 模型。以下是示例的主要介绍：

### 如何将基础模型修改为并行处理版本

要将基础模型修改为并行处理版本，主要需要完成以下步骤：

1. **引入并行处理模块**  
   在并行版本中，使用了 `ParallelProcessor` 类来管理分析和处理任务的并行执行。需要在代码中引入该模块，并替换原有的顺序处理逻辑。

   ```python
   from jittor_geometric.dygstore.parallel import ParallelProcessor
   ```

2. **实现数据分块**  
   在并行版本中，数据需要被分割为多个块（chunks），以便在多个线程或进程中并行处理。可以通过 `split_data_into_chunks` 方法实现数据分块。

   ```python
   def split_data_into_chunks(self, data, chunk_size=50):
       total_events = len(data[0])
       chunks = []
       for i in range(0, total_events, chunk_size):
           chunk_data = tuple(d[i:i+chunk_size] for d in data)
           chunks.append(chunk_data)
       return chunks
   ```

3. **实现分析和处理流水线**  
   使用 `ParallelProcessor` 的 `execute_pipeline` 方法，将分析和处理逻辑分离，并通过队列实现数据的传递。

   ```python
   results = parallel_processor.execute_pipeline(
       analyzer_func=analyzer_func,
       processor_func=processor_func,
       dyrep_obj=self,
       chunks=chunks
   )
   ```

4. **实现事件级并行处理**  
   在事件处理部分，使用 `process_events_parallel` 方法对每个事件进行并行处理。

   ```python
   event_results = parallel_processor.process_events_parallel(
       process_event_func=process_event_func,
       batch_size=batch_size
   )
   ```

5. **修改模型的执行逻辑**  
   在并行版本中，`execute` 方法需要调用上述并行处理逻辑，并对结果进行合并。

   ```python
   def execute(self, data):
       chunks = self.split_data_into_chunks(data, chunk_size=50)
       results = parallel_processor.execute_pipeline(
           analyzer_func=analyzer_func,
           processor_func=processor_func,
           dyrep_obj=self,
           chunks=chunks
       )
       return self._merge_results(results)
   ```

### 示例代码对比

以下是基础版本和并行版本的主要区别：

#### 基础版本（`dyrepbase.py`）

```python
def execute(self, data):
    u, v, time_diff, event_types, t_bar, t = data[:6]
    for it in range(len(u)):
        u_it, v_it, et_it, td_it = u[it], v[it], event_types[it], time_diff[it]
        lambda_uv_it = self.compute_intensity_lambda(self.z[u_it], self.z[v_it], et_it)
        z_new = self.update_node_embedding(self.z, u_it, v_it, td_it)
        self.update_A_S(u_it, v_it, et_it, lambda_uv_it)
        self.z = z_new
```

#### 并行版本（`dyrepupdate.py`）

```python
def execute(self, data):
    chunks = self.split_data_into_chunks(data, chunk_size=50)
    parallel_processor = ParallelProcessor(
        max_analyzer_threads=1,
        max_processor_threads=1,
        max_event_workers=1,
        queue_maxsize=5,
        use_processes=False
    )
    results = parallel_processor.execute_pipeline(
        analyzer_func=analyzer_func,
        processor_func=processor_func,
        dyrep_obj=self,
        chunks=chunks
    )
    return self._merge_results(results)
```

通过以上步骤，可以将基础版本的 DyRep 模型修改为支持并行处理的版本。
