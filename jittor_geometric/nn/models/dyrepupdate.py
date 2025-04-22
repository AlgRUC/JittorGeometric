import numpy as np
import sys
import jittor
from datetime import datetime, timedelta
from jittor.nn import Linear, ModuleList, Parameter, init
from jittor_geometric.dygstore.adjlist import DiAdjList, TEdge, EdgeData
from jittor_geometric.dygstore.embedding import DyNodeEmbedding
from jittor_geometric.dygstore.dependedgraph import DependedGraph
from jittor_geometric.dygstore.dysubgraph import DySubGraph
from jittor_geometric.dygstore.parallel import ParallelProcessor
from concurrent.futures import ThreadPoolExecutor,as_completed
import threading
import time
from concurrent.futures import ProcessPoolExecutor  # 替换ThreadPoolExecutor
import multiprocessing as mp
from multiprocessing import Process
from queue import Queue
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyzer_func(chunks, dyrep_obj, analyze_to_process_queue):
    # logger.info(f"Starting analyzer with {len(chunks)} chunks")
    for chunk in chunks:
        # logger.info(f"Processing chunk with {len(chunk[0])} events")
        subgraphs = dyrep_obj._create_subgraphs_from_data(chunk)
        dep_graph = DependedGraph(total_events=len(chunk[0]), start_id=0)
        dep_graph.analyze_batch(subgraphs)
        batches = []
        while True:
            batch = dep_graph.get_parallel_batch()
            # logger.info(f"Got batch: {batch}")
            if batch is None:
                break
            if not batch:
                continue
            batch_indices = batch.tolist() if isinstance(batch, jittor.Var) else batch
            batches.append(batch_indices)
        # logger.info(f"Putting {len(batches)} batches into queue")
        analyze_to_process_queue.put(batches)
    # logger.info("Analyzer sending None to signal completion")
    analyze_to_process_queue.put(None)
    # logger.info("Analyzer completed")

def processor_func(dyrep_obj, analyze_to_process_queue, process_results_queue, chunks):
    # logger.info("Starting processor")
    all_results = []
    idx = 0
    while True:
        # logger.info("Waiting for queue item")
        item = analyze_to_process_queue.get()
        if item is None:
            # logger.info("Received None, exiting processor")
            break
        batches = item
        # logger.info(f"Processing {len(batches)} batches for chunk {idx}")
        chunk = chunks[idx]
        for batch_indices in batches:
            batch_data = dyrep_obj._get_batch_data(chunk, batch_indices)
            results = dyrep_obj._process_batch(batch_data)
            all_results.append(results)
        idx += 1
    # logger.info("Putting results into results queue")
    process_results_queue.put(all_results)
    # logger.info("Processor completed")

class DyRep(jittor.nn.Module):
    def __init__(self, num_nodes, hidden_dim, random_state, first_date, end_datetime, num_neg_samples=5, num_time_samples=10,
                 device='cpu', all_comms=False, train_td_max=None):
        super(DyRep, self).__init__()

        self.dep_graph = DependedGraph(
            total_events=200,  
            start_id=0
        )
        self.batch_update = False
        self.all_comms = all_comms
        self.include_link_features = False

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.random_state = random_state
        self.first_date = first_date
        self.end_datetime = end_datetime
        self.num_neg_samples = num_neg_samples
        self.device = device
        self.num_time_samples = num_time_samples
        self.n_assoc_types = 1
        self.train_td_max = train_td_max
        self.elapsed_time = 0
        
        if not self.include_link_features:
            self.omega = ModuleList([Linear(in_features=2*hidden_dim, out_features=1),
                                     Linear(in_features=2*hidden_dim, out_features=1)])
        else:
            self.omega = ModuleList([Linear(in_features=2*hidden_dim+172, out_features=1),
                                     Linear(in_features=2*hidden_dim+172, out_features=1)])
        self.psi = jittor.nn.Parameter(0.5 * jittor.ones(2))  # type=2: assoc + comm-

        self.W_h = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W_struct = Linear(in_features=hidden_dim*self.n_assoc_types, out_features=hidden_dim)
        self.W_rec = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W_t = Linear(4, hidden_dim)  # [days, hours, minutes, seconds]

    def reset_state(self, node_embeddings_initial, dygraph, node_degree_initial, 
                    time_bar, resetS=False):
        #Initialize the state, using dygraph to replace the original A_initial
        self.dz = DyNodeEmbedding()
        self.dz.inits(node_embeddings_initial)
        # Initialize node embeddings (originally used z, now uniformly changed to dz)
        self.dygraph = dygraph  # DiAdjList对象
        self.node_degree_global = node_degree_initial
        self.time_bar = time_bar

        self.initialize_S_from_dygraph()  

        self.Lambda_dict = jittor.zeros(5000).to(self.device)
        self.time_keys = []

    def initialize_S_from_dygraph(self):
        self.S = jittor.zeros((self.num_nodes, self.num_nodes, self.n_assoc_types))
        
        for at in range(self.n_assoc_types):
            for u in range(self.num_nodes):
                out_neighbors = self.dygraph.out_neighbors(u)
                degree = len(out_neighbors)
                if degree > 0:
                    for v in out_neighbors:
                        self.S[u, v, at] = 1.0 / degree

    def _get_batch_data(self, data, batch_indices):
        """ 
        Args:
            data: Original data tuple with structure (u, v, time_diff, event_types, t_bar, t, ...)
            batch_indices: List[int] or jittor.Var, indices of events to extract
        
        return:
            batch_data: Sub-batch data tuple with the same structure as the original
        """
        # Convert indices uniformly to jittor.Var format (with list input compatibility)
        if isinstance(batch_indices, list):
            batch_indices = jittor.array(batch_indices).long()
        
        # Unpack the original data structure (according to data definition in DyRep.execute)
        u = data[0]  # shape: [total_events]
        v = data[1]
        time_diff = data[2]  # shape: [total_events, 4]
        event_types = data[3] # shape: [total_events]
        t_bar = data[4]      # shape: [total_events, num_nodes] 
        t = data[5]          # shape: [total_events]
        
        # Perform slicing operation on each data component
        def slice_tensor(tensor):
            if tensor.ndim == 1:
                return tensor[batch_indices]
            elif tensor.ndim == 2:
                return tensor[batch_indices, :]
            elif tensor.ndim == 3:
                return tensor[batch_indices, :, :]
            else:
                raise ValueError(f"Unsupported tensor dimension: {tensor.ndim}")

        sliced_data = (
            slice_tensor(u),
            slice_tensor(v),
            slice_tensor(time_diff),
            slice_tensor(event_types),
            slice_tensor(t_bar), 
            slice_tensor(t)
        )
        
        return sliced_data

    def split_data_into_chunks(self, data, chunk_size=50):
        """Split the data into multiple chunks"""
        total_events = len(data[0])
        chunks = []
        for i in range(0, total_events, chunk_size):
            chunk_data = tuple(d[i:i+chunk_size] for d in data)
            chunks.append(chunk_data)
        return chunks
   
    def execute(self, data):
        # Split data into chunks
        chunks = self.split_data_into_chunks(data, chunk_size=50)
        # Initialize the parallel processor
        parallel_processor = ParallelProcessor(
            max_analyzer_threads=1,
            max_processor_threads=1,
            max_event_workers=1,  # Adjust based on system resources
            queue_maxsize=5,
            use_processes=False  # Set to True for ProcessPoolExecutor
        )

        # Execute the pipeline
        results = parallel_processor.execute_pipeline(
            analyzer_func=analyzer_func,
            processor_func=processor_func,
            dyrep_obj=self,
            chunks=chunks
        )

        # Merge results
        return self._merge_results(results)


    def g_fn(self, dz_cat, k, edge_type=None, dz2=None):
        if dz2 is not None:
            dz_cat = jittor.cat((dz_cat, dz2), dim=1)
        else:
            raise NotImplementedError('')

        g = dz_cat.new(len(dz_cat), 1).fill_(0)
        idx = k <= 0
        if jittor.sum(idx) > 0:
            if edge_type is not None:
                dz_cat1 = jittor.cat((dz_cat[idx], edge_type[idx, :self.n_assoc_types]), dim=1)
            else:
                dz_cat1 = dz_cat[idx]
            g[idx] = self.omega[0](dz_cat1)
        idx = k > 0
        if jittor.sum(idx) > 0:
            if edge_type is not None:
                dz_cat1 = jittor.cat((dz_cat[idx], edge_type[idx, self.n_assoc_types:]), dim=1)
            else:
                dz_cat1 = dz_cat[idx]
            g[idx] = self.omega[1](dz_cat1)

        g = g.flatten()
        return g

    def compute_intensity_lambda(self, dz_u, dz_v, et_uv):
        dz_u = dz_u.view(-1, self.hidden_dim)
        dz_v = dz_v.view(-1, self.hidden_dim)
        dz_cat = jittor.cat((dz_u, dz_v), dim=1)
        et = (et_uv > 0).long()
        g = dz_cat.new_zeros(len(dz_cat))
        for k in range(2):
            idx = (et == k) 
            if jittor.sum(idx) > 0:
                g[idx] = self.omega[k](dz_cat).flatten()[idx]
        psi = self.psi[et]
        g_psi = jittor.clamp(g / (psi + 1e-7), -75, 75)
        Lambda = psi * jittor.log(1 + jittor.exp(g_psi))
        return Lambda

    def update_node_embedding(self, u, v, td):
        """Update node embeddings using dygraph to capture structural information"""
        dz_prev = self.dz.m_embedding[0]
        z_new = dz_prev.clone()
        h_u_struct = jittor.zeros((2, self.hidden_dim, self.n_assoc_types))

        for cnt, (target, source) in enumerate([(u, v), (v, u)]):
            neighbors = self.dygraph.out_neighbors(source)
            if len(neighbors) > 0:
                neighbor_embeddings = dz_prev[neighbors]
                h_i_bar = self.W_h(neighbor_embeddings)
                weights = jittor.exp(self.S[source, neighbors, 0])
                weights /= jittor.sum(weights)
                h_u_struct[cnt, :, 0] = jittor.sum(weights.view(-1, 1) * h_i_bar, dim=0)

        z_new[[u, v]] = jittor.sigmoid(
            self.W_struct(h_u_struct.view(2, -1)) +
            self.W_rec(dz_prev[[u, v]]) +
            self.W_t(td).view(2, -1)
        )
        return z_new

    def update_dygraph_S(self, u, v, event_type, lambda_t):
        if event_type == 0:
            edge = TEdge(src=u, dst=v, edge_data=EdgeData(value=1.0))
            self.dygraph.add_edge(edge)
            edge = TEdge(src=v, dst=u, edge_data=EdgeData(value=1.0))
            self.dygraph.add_edge(edge)
        for direction in [(u, v), (v, u)]:
            j, i = direction
            for rel in range(self.n_assoc_types):
                degree_j = self.dygraph.out_degree(j)
                b = 1.0 / (degree_j + 1e-7) if degree_j > 0 else 0
                if self.dygraph.has_edge(j, i):
                    self.S[j, i, rel] = b + lambda_t
                row_sum = jittor.sum(self.S[j, :, rel])
                self.S[j, :, rel] /= (row_sum + 1e-7)

    def compute_cond_density(self, u, v, time_bar):
        N = self.num_nodes
        s_uv = self.Lambda_dict.new_zeros((2, N))
        Lambda_sum = jittor.cumsum(self.Lambda_dict.flip(0), 0).flip(0) / len(self.Lambda_dict)
        time_keys_min = self.time_keys[0]
        time_keys_max = self.time_keys[-1]

        indices = []
        l_indices = []
        t_bar_min = jittor.min(time_bar[[u, v]]).item()
        if t_bar_min < time_keys_min:
            start_ind_min = 0
        elif t_bar_min > time_keys_max:
            return s_uv
        else:
            start_ind_min = self.time_keys.index(int(t_bar_min))

        max_pairs = jittor.max(jittor.cat((time_bar[[u, v]].view(1, 2).expand(N, -1).t().contiguous().view(2 * N, 1),
                                          time_bar.repeat(2, 1)), dim=1), dim=1)
        max_pairs = max_pairs.view(2, N).long().numpy()

        for c, j in enumerate([u, v]):
            for i in range(N):
                if i == j:
                    continue
                t_bar_val = max_pairs[c, i]
                if t_bar_val < time_keys_min:
                    start_ind = 0
                elif t_bar_val > time_keys_max:
                    continue
                else:
                    start_ind = self.time_keys.index(t_bar_val, start_ind_min)
                indices.append((c, i))
                l_indices.append(start_ind)

        indices = np.array(indices)
        l_indices = np.array(l_indices)
        s_uv[indices[:, 0], indices[:, 1]] = Lambda_sum[l_indices]

        return s_uv

    def _create_subgraphs_from_data(self, data):
        """Convert raw data into a list of DySubGraph objects"""
        u, v, time_diff, event_types, t_bar, t = data[:6]
        subgraphs = []
        for i in range(len(u)):
            subg = DySubGraph(
                u=int(u[i]),
                v=int(v[i]),
                event_type=int(event_types[i]),
                timestamp=datetime.fromtimestamp(t[i]),
                di_adjlist=self.dygraph,  # Share the same graph structure
                hop=1
            )
            subgraphs.append(subg)
        return subgraphs
    
    def _process_event(self, it, u_all, v_all, event_types, time_diff, t_bar, t):
        # Existing process_event logic, adjusted to take parameters explicitly
        u_it = int(u_all[it])
        v_it = int(v_all[it])
        et_it = event_types[it]
        td_it = time_diff[it]

        event_result = {'u': u_it, 'v': v_it, 'et': et_it}
        dz_prev = self.dz.m_embedding[0]

        lambda_uv_it = self.compute_intensity_lambda(self.dz.index(u_it), self.dz.index(v_it), et_it)
        event_result['lambda_uv'] = lambda_uv_it

        z_new = self.update_node_embedding(u_it, v_it, td_it)
        assert jittor.sum(jittor.isnan(z_new)) == 0, (jittor.sum(jittor.isnan(z_new)), z_new, it)
        if not self.batch_update:
            self.update_dygraph_S(u_it, v_it, et_it, lambda_uv_it.item())
            for j in [u_it, v_it]:
                self.node_degree_global[0][j] = self.dygraph.out_degree(j)
        
        batch_nodes = np.delete(np.arange(self.num_nodes), [u_it, v_it])
        batch_uv_neg = self.random_state.choice(batch_nodes, size=self.num_neg_samples * 2,
                                            replace=(len(batch_nodes) < 2 * self.num_neg_samples))
        neg_emb_u = jittor.cat((dz_prev[u_it].expand(self.num_neg_samples, -1),
                                dz_prev[batch_uv_neg[self.num_neg_samples:]]), dim=0)
        neg_emb_v = jittor.cat((dz_prev[batch_uv_neg[:self.num_neg_samples]],
                                dz_prev[v_it].expand(self.num_neg_samples, -1)), dim=0)
        event_result['neg_emb_u'] = neg_emb_u
        event_result['neg_emb_v'] = neg_emb_v
        
        with jittor.no_grad():
            dz_uv_it = jittor.cat((dz_prev[u_it].unsqueeze(0).expand(self.num_nodes, -1),
                                dz_prev[v_it].unsqueeze(0).expand(self.num_nodes, -1)), dim=0)
            lambda_uv_pred = self.compute_intensity_lambda(dz_uv_it, dz_prev.repeat(2, 1), et_it.repeat(len(dz_uv_it)))
            event_result['t'] = int(t[it])
            event_result['t_bar'] = t_bar[it].clone()
            event_result['lambda_uv_pred'] = lambda_uv_pred

            if not self.training:
                t_cur_date = datetime.fromtimestamp(int(t[it]))
                t_prev = datetime.fromtimestamp(int(max(t_bar[it][u_it], t_bar[it][v_it])))
                td = t_cur_date - t_prev
                time_scale_hour = round((td.days * 24 + td.seconds / 3600), 3)
                delta = np.array([td.days, td.seconds // 3600, (td.seconds // 60) % 60, td.seconds % 60], float)
                delta = np.tile(delta, (2, 1))
                delta = jittor.array(delta).float().to(self.device)
                prev_embedding = dz_prev.clone()
                surv_allsamples = prev_embedding.new_zeros(self.num_time_samples)
                factor_samples = 2 * self.random_state.rand(self.num_time_samples)
                sampled_time_scale = time_scale_hour * factor_samples

                embeddings_u_samples = []
                embeddings_v_samples = []
                for n in range(1, self.num_time_samples + 1):
                    embeddings_u_samples.append(prev_embedding[u_it])
                    embeddings_v_samples.append(prev_embedding[v_it])

                    td_sample = timedelta(hours=sampled_time_scale[n - 1])
                    delta_sample = np.array([td_sample.days, td_sample.seconds // 3600,
                                            (td_sample.seconds // 60) % 60, td_sample.seconds % 60], float)
                    delta_sample = jittor.array(np.tile(delta_sample, (2, 1))).float().to(self.device)

                    new_embedding = self.update_node_embedding(u_it, v_it, delta_sample)

                    batch_uv_neg_sample = self.random_state.choice(batch_nodes, size=self.num_neg_samples * 2,
                                                                replace=(len(batch_nodes) < 2 * self.num_neg_samples))
                    u_neg_sample = batch_uv_neg_sample[self.num_neg_samples:]
                    v_neg_sample = batch_uv_neg_sample[:self.num_neg_samples]
                    embeddings_u_neg = jittor.cat((prev_embedding[u_it].view(1, -1).expand(self.num_neg_samples, -1),
                                                prev_embedding[u_neg_sample]), dim=0)
                    embeddings_v_neg = jittor.cat((prev_embedding[v_neg_sample],
                                                prev_embedding[v_it].view(1, -1).expand(self.num_neg_samples, -1)), dim=0)

                    surv_sample = (sum(self.compute_intensity_lambda(embeddings_u_neg, embeddings_v_neg,
                                                                    jittor.zeros(len(embeddings_u_neg)))) +
                                sum(self.compute_intensity_lambda(embeddings_u_neg, embeddings_v_neg,
                                                                    jittor.ones(len(embeddings_u_neg)))))
                    surv_allsamples[n - 1] = surv_sample / self.num_neg_samples
                    prev_embedding = new_embedding

                embeddings_u_samples = jittor.stack(embeddings_u_samples, dim=0)
                embeddings_v_samples = jittor.stack(embeddings_v_samples, dim=0)
                lambda_t_allsamples = self.compute_intensity_lambda(embeddings_u_samples, embeddings_v_samples,
                                                                jittor.zeros(self.num_time_samples) + et_it)
                f_samples = lambda_t_allsamples * jittor.exp(-surv_allsamples)
                expectation = jittor.array(np.cumsum(sampled_time_scale)) * f_samples
                event_result['expected_time'] = expectation.sum() / self.num_time_samples
            else:
                event_result['expected_time'] = None
        self.dz.update(u_it, z_new[u_it], dim=0)
        self.dz.update(v_it, z_new[v_it], dim=0)

        return event_result

    def _process_batch(self, batch_data):
        # Unpack data
        u, v, time_diff, event_types, t_bar, t = batch_data[:6]
        batch_size = len(u)
        u_all, v_all = u.data, v.data

        # Initialize A_pred and surv for non-training mode
        A_pred, surv = None, None
        if not self.training:
            A_pred = jittor.zeros((batch_size, self.num_nodes, self.num_nodes))
            surv = jittor.zeros((batch_size, self.num_nodes, self.num_nodes))

        # Normalize time differences
        time_mean = jittor.array(np.array([0, 0, 0, 0])).float().to(self.device).view(1, 1, 4)
        time_sd = jittor.array(np.array([50, 7, 15, 15])).float().to(self.device).view(1, 1, 4)
        time_diff = (time_diff - time_mean) / time_sd

        # Initialize lists for results
        lambda_uv_list = []
        batch_embeddings_u_neg = []
        batch_embeddings_v_neg = []
        expected_time = []
        batch_events = []

        # Create a lock for shared resources
        # lock = threading.Lock()

        # Initialize ParallelProcessor
        parallel_processor = ParallelProcessor(
            max_event_workers=1,  # Adjust based on system resources
            use_processes=False  # Set to True for ProcessPoolExecutor
        )

        # Process events in parallel
        results = parallel_processor.process_events_parallel(
            process_event_func=lambda idx: self._process_event(idx, u_all, v_all, event_types, time_diff, t_bar, t),
            batch_size=batch_size
        )

        # Collect results
        for res in results:
            lambda_uv_list.append(res['lambda_uv'])
            if res.get('expected_time') is not None:
                expected_time.append(res['expected_time'])
            batch_embeddings_u_neg.append(res['neg_emb_u'])
            batch_embeddings_v_neg.append(res['neg_emb_v'])
            batch_events.append({
                'u': res['u'],
                'v': res['v'],
                't': res['t'],
                'lambda_uv_pred': res['lambda_uv_pred'],
                't_bar': res['t_bar']
            })

        # Aggregate results
        lambda_uv = jittor.cat(lambda_uv_list, dim=0)
        batch_embeddings_u_neg = jittor.cat(batch_embeddings_u_neg, dim=0)
        batch_embeddings_v_neg = jittor.cat(batch_embeddings_v_neg, dim=0)
        lambda_uv_neg_0 = self.compute_intensity_lambda(batch_embeddings_u_neg, batch_embeddings_v_neg, jittor.Var(0))
        lambda_uv_neg_1 = self.compute_intensity_lambda(batch_embeddings_u_neg, batch_embeddings_v_neg, jittor.Var(1))
        lambda_uv_neg = (lambda_uv_neg_0 + lambda_uv_neg_1) / self.num_neg_samples

        return lambda_uv, lambda_uv_neg, batch_events, expected_time

    def _merge_results(self, results):
        # Unpack batch results
        lambda_uv_list = [r[0] for r in results]
        lambda_uv_neg_list = [r[1] for r in results]
        all_events_data = [r[2] for r in results]  # Event data from all batches
        expected_time_list = [r[3] for r in results]

        # === Merge all events and sort by timestamp ===
        merged_events = []
        for batch_events in all_events_data:
            merged_events.extend(batch_events)
        merged_events = sorted(merged_events, key=lambda x: x['t'])  # Sort by timestamp

        # === Initialize A_pred and surv for non-training mode ===
        A_pred, surv = None, None
        if not self.training:
            total_events = len(merged_events)
            A_pred = jittor.zeros((total_events, self.num_nodes, self.num_nodes))
            surv = jittor.zeros((total_events, self.num_nodes, self.num_nodes))

        for global_it, event in enumerate(merged_events):
            u_it = event['u']
            v_it = event['v']
            lambda_uv_pred = event['lambda_uv_pred']
            t_bar_it = event['t_bar']
            time_key = event['t']

            # Update A_pred and surv (non-training mode)
            if not self.training:
                A_pred[global_it, u_it, :] = lambda_uv_pred[:self.num_nodes]
                A_pred[global_it, v_it, :] = lambda_uv_pred[self.num_nodes:]
                s_u_v = self.compute_cond_density(u_it, v_it, t_bar_it)
                surv[global_it, [u_it, v_it], :] = s_u_v

            # Update Lambda_dict and time_keys
            idx = np.delete(np.arange(self.num_nodes), [u_it, v_it])
            idx = np.concatenate((idx, idx + self.num_nodes))
            
            # Rolling update logic
            if len(self.time_keys) >= len(self.Lambda_dict):
                time_keys = np.array(self.time_keys)
                time_keys[:-1] = time_keys[1:]
                self.time_keys = list(time_keys[:-1])
                self.Lambda_dict[:-1] = self.Lambda_dict.clone()[1:]
                self.Lambda_dict[-1] = 0
            
            self.Lambda_dict[len(self.time_keys)] = lambda_uv_pred[idx].sum()
            self.time_keys.append(time_key)

        # Merge other results
        final_lambda = jittor.concat(lambda_uv_list) if lambda_uv_list else jittor.zeros(0)
        final_neg = jittor.concat(lambda_uv_neg_list) if lambda_uv_neg_list else jittor.zeros(0)
        flat_expected_time = [item for sublist in expected_time_list for item in sublist]

        return final_lambda, final_neg, A_pred, surv, flat_expected_time

    def reset_dep_graph(self):
        # Reconstruct a new DependedGraph object according to requirements
        self.dep_graph = DependedGraph(total_events=200, start_id=0)
