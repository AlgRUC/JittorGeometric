import platform

import numpy as np
import sys
import os
import time
import copy
import pickle
import jittor.nn as nn
from datetime import datetime

from sklearn.metrics import average_precision_score, roc_auc_score
from jittor.dataset import DataLoader
import jittor.optim as optim
import argparse
import matplotlib.pyplot as plt
import jittor
from jittor import lr_scheduler

from jittor_geometric.datasets import JODIEDataset, TemporalDataLoader, GithubDataset
from jittor_geometric.nn.models.dyrepupdate import DyRep
from tqdm import tqdm
from collections import defaultdict

from jittor_geometric.dygstore.adjlist import DiAdjList, TEdge, EdgeData
from jittor_geometric.dygstore.embedding import InitialEmb,DyNodeEmbedding

def get_return_time(data_set):
    reoccur_dict = {}
    for n1,n2,r,t in data_set.all_events:
        et = 0 if r in data_set.assoc_types else 1
        key = (n1,n2,et) if n1<=n2 else (n2,n1,et)
        ts = t.timestamp()
        if key not in reoccur_dict:
            reoccur_dict[key] = [ts]
        elif ts ==  reoccur_dict[key][-1]:
            continue
        else:
            reoccur_dict[key].append(ts)
    count = 0
    for event in reoccur_dict:
        occ = reoccur_dict[event]
        if len(occ) > 1:
            count += len(occ)-1
    print("Number of repeat events in the test set: {}".format(count))
    reoccur_time_ts = np.zeros(len(data_set.all_events))
    reoccur_time_hr = np.zeros(len(data_set.all_events))
    for idx, (n1,n2,r,t) in enumerate(data_set.all_events):
        et = 0 if r in data_set.assoc_types else 1
        key = (n1,n2,et) if n1<=n2 else (n2,n1,et)
        val = reoccur_dict[key]
        if len(val) == 1 or t.timestamp()==val[-1]:
            reoccur_time_ts[idx] = data_set.END_DATE.timestamp()
            reoccur_time = data_set.END_DATE - t
            reoccur_time_hr[idx] = round((reoccur_time.days*24 + reoccur_time.seconds/3600),3)
        else:
            reoccur_time_ts[idx] = val[val.index(t.timestamp()) + 1]
            reoccur_time = datetime.fromtimestamp(int(reoccur_time_ts[idx])) - t
            reoccur_time_hr[idx] = round((reoccur_time.days*24 + reoccur_time.seconds/3600),3)
    return reoccur_dict, reoccur_time_ts, reoccur_time_hr

def mae_error(u, v, k, time_cur, expected_time, reoccur_dict, end_date):
    u, v, time_cur = u.data.cpu().numpy(), v.data.cpu().numpy(), time_cur.data.cpu().numpy()
    et = (k>0).int().data.cpu().numpy()
    batch_predict_time = []
    N = len(u)
    ae = 0
    for idx in range(N):
        key = (u[idx], v[idx], et[idx]) if u[idx] <= v[idx] else (v[idx], u[idx], et[idx])
        val = reoccur_dict[key]
        td_pred_hour = expected_time[idx]
        if len(val) == 1 or time_cur[idx]==val[-1]:
            next_ts = end_date.timestamp()
        else:
            next_ts = val[val.index(time_cur[idx])+1]
        true_td = datetime.fromtimestamp(int(next_ts))-datetime.fromtimestamp(int(time_cur[idx]))
        td_true_hour = round((true_td.days*24 + true_td.seconds/3600), 3)
        ae += abs(td_pred_hour-td_true_hour)
        batch_predict_time.append((td_pred_hour, td_true_hour))
    return ae, batch_predict_time

def MAE(expected_time_hour, batch_ts_true, t_cur):
    t_cur = t_cur.data.cpu().numpy()
    valid_idx = np.where(batch_ts_true != 0)
    t_cur_dt = np.array(list(map(lambda x: datetime.fromtimestamp(int(x)), t_cur[valid_idx])))
    batch_dt_true = np.array(list(map(lambda x: datetime.fromtimestamp(int(x)), batch_ts_true[valid_idx])))
    batch_time_true = batch_dt_true - t_cur_dt
    batch_time_hour_true = np.array(list(map(lambda td: round(td.days * 24 + td.seconds/3600, 3), batch_time_true)))
    expected_time_hour = np.array(expected_time_hour)[valid_idx]
    batch_ae = sum(abs(expected_time_hour-batch_time_hour_true))
    batch_res = list(zip(expected_time_hour, batch_time_hour_true))
    return batch_ae, batch_res

def test_time_pred(model, reoccur_dict, reoccur_time_hr):
    model.eval()
    loss = 0
    losses =[ [np.Inf, 0], [np.Inf, 0] ]

    total_ae, total_sample_num = 0, 0.000001
    # all_res = []
    # test_loader.dataset.time_bar = np.zeros((test_loader.dataset.N_nodes, 1)) + test_loader.dataset.FIRST_DATE.timestamp()
    end_date = test_loader.dataset.END_DATE
    with jittor.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            data[2] = data[2].float().to(args.device)
            data[4] = data[4].double().to(args.device)
            data[5] = data[5].double()
            batch_size = len(data[0])
            output = model(data)

            loss += (-jittor.sum(jittor.log(output[0]) + 1e-10) + jittor.sum(output[1])).item()
            for i in range(len(losses)):
                m1 = output[i].min()
                m2 = output[i].max()
                if m1 < losses[i][0]:
                    losses[i][0] = m1
                if m2 > losses[i][1]:
                    losses[i][1] = m2
            u, v, k, time_cur = data[0], data[1], data[3], data[5]
            ########### use the event sequence to predict time
            # if batch_idx == 0:
            #     start_time = time_cur[0]
            # ae, batch_pred_res = time_prediction_error(A_pred, u, v, k, time_cur, start_time, Survival_term, reoccur_dict)
            ########### predict with repeat current event sequence with reoccur_dict
            ae, batch_pred_res = mae_error(u,v,k,time_cur,output[-1],reoccur_dict, end_date)
            ########### predict with repeat current event sequence with reoccur_time_true
            # ae, batch_pred_res = MAE(output[-1], reoccur_time_true[batch_idx*batch_size:(batch_idx+1)*batch_size], time_cur)
            ###########
            total_ae += ae
            total_sample_num += len(batch_pred_res)
            if batch_idx % 20 == 0:
                print('\nTEST batch={}/{}, time prediction MAE {}, loss {:.3f}'.
                      format(batch_idx + 1, len(test_loader), (total_ae / total_sample_num),
                             (loss / ((batch_idx + 1)*batch_size))))

    print('\nTEST batch={}/{}, time prediction MAE {}, loss={:.3f}, loss_event min/max={:.4f}/{:.4f}, '
          'loss_nonevent min/max={:.4f}/{:.4f}'.
          format(batch_idx + 1, len(test_loader), (total_ae/total_sample_num), (loss / len(test_loader.dataset)),
                 losses[0][0], losses[0][1], losses[1][0], losses[1][1],
                 len(model.Lambda_dict), time_iter / (batch_idx + 1)))
    return total_ae/total_sample_num, loss/len(test_loader.dataset)


def test_all(model, reoccur_dict, return_time_hr):
    model.eval()
    model.reset_dep_graph()
    print(model.training)
    loss = 0
    total_ae= 0
    # test_loader.dataset.time_bar = np.zeros(
    #     (test_loader.dataset.N_nodes, 1)) + test_loader.dataset.FIRST_DATE.timestamp()
    aps, aucs = [], []
    with jittor.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            data[2] = data[2].float().to(args.device)
            data[4] = data[4].double().to(args.device)
            data[5] = data[5].double()
            batch_size = len(data[0])
            output = model(data)
            A_pred, Survival_term = output[2], output[3]
            cond = A_pred * jittor.exp(-Survival_term)

            loss += (-jittor.sum(jittor.log(output[0]) + 1e-10) + jittor.sum(output[1])).item()
            u, v, k, time_cur = data[0], data[1], data[3], data[5]

            neg_v_all = np.delete(np.arange(train_set.N_nodes), jittor.cat([u, v]).cpu().numpy())
            neg_v = jittor.Var(rnd.choice(neg_v_all, size=batch_size, replace=len(neg_v_all) < batch_size))
                                 
            pos_prob = cond[np.arange(batch_size), u, v]
            neg_prob = cond[np.arange(batch_size), u, neg_v]
            y_pred = jittor.cat([pos_prob, neg_prob], dim=0).cpu()
            y_true = jittor.cat(
                [jittor.ones(pos_prob.size(0)),
                 jittor.zeros(neg_prob.size(0))], dim=0)
            ap = average_precision_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            aps.append(ap)
            aucs.append(auc)
            return_time_pred = jittor.stack(output[-1]).cpu().numpy()
            mae = np.mean(abs(return_time_pred - return_time_hr[batch_idx*args.batch_size:(batch_idx*200+batch_size)]))
            total_ae += mae * batch_size
            if batch_idx % 20 == 0:
                print('\nTEST batch={}/{}, time prediction MAE {}, loss {:.3f}, ap {}, auc {}'.
                      format(batch_idx + 1, len(test_loader), mae,
                             (loss / ((batch_idx + 1) * batch_size)), ap, auc))
    return total_ae / len(test_set.all_events), loss / len(test_set.all_events), \
           float(jittor.Var(aps).mean()), float(jittor.Var(aucs).mean())

def test(model, reoccur_dict, n_test_batches=None):
    model.eval()
    loss = 0
    losses =[ [np.Inf, 0], [np.Inf, 0] ]
    n_samples = 0
    # Time slots with 10 days intervals as in the DyRep paper
    timeslots = [t.toordinal() for t in test_loader.dataset.TEST_TIMESLOTS]
    end_date = test_set.END_DATE
    event_types = list(test_loader.dataset.event_types_num.keys()) #['comm', 'assoc']
    # sort it by k
    for event_t in test_loader.dataset.event_types_num:
        event_types[test_loader.dataset.event_types_num[event_t]] = event_t

    ## Com means the communication event type (will not change the network structure)
    event_types += ['Com']
    total_ae, total_sample_num = 0, 0.000001
    mar, hits_10 = {}, {}
    for event_t in event_types:
        mar[event_t] = []
        hits_10[event_t] = []
        for c, slot in enumerate(timeslots):
            mar[event_t].append([])
            hits_10[event_t].append([])

    start = time.time()

    with jittor.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            data[2] = data[2].float().to(args.device)
            data[4] = data[4].double().to(args.device)
            data[5] = data[5].double()
            batch_size = len(data[0])
            output = model(data)

            loss += (-jittor.sum(jittor.log(output[0]) + 1e-10) + jittor.sum(output[1])).item()
            for i in range(len(losses)):
                m1 = output[i].min()
                m2 = output[i].max()
                if m1 < losses[i][0]:
                    losses[i][0] = m1
                if m2 > losses[i][1]:
                    losses[i][1] = m2
            n_samples += 1
            A_pred, Survival_term = output[2], output[3]
            u, v, k, time_cur = data[0], data[1], data[3], data[5]

            ae, batch_pred_res = mae_error(u, v, k, time_cur, output[-1], reoccur_dict, end_date)

            total_ae += ae
            total_sample_num += len(batch_pred_res)

            m, h = MAR(A_pred, u, v, k, Survival_term=Survival_term)
            assert len(time_cur) == len(m) == len(h) == len(k)
            for t, m, h, k_ in zip(time_cur, m, h, k):
                d = datetime.fromtimestamp(t.item()).toordinal()
                event_t = event_types[k_.item()]
                for c, slot in enumerate(timeslots):
                    if d <= slot:
                        mar[event_t][c].append(m)
                        hits_10[event_t][c].append(h)
                        if k_ > 0:
                            mar['Com'][c].append(m)
                            hits_10['Com'][c].append(h)
                        if c > 0:
                            assert slot > timeslots[c-1] and d > timeslots[c-1], (d, slot, timeslots[c-1])
                        break

            if batch_idx % 20 == 0:
                print('\nTEST batch={}/{}, time prediction MAE {}, loss={:.3f}'.
                      format(batch_idx + 1, len(test_loader), (total_ae / total_sample_num),
                             (loss / ((batch_idx + 1)*batch_size))))

            if n_test_batches is not None and batch_idx >= n_test_batches - 1:
                break

    time_iter = time.time() - start


    print('\nTEST batch={}/{}, time prediction MAE {}, loss={:.3f}, psi={}, loss1 min/max={:.4f}/{:.4f}, '
          'loss2 min/max={:.4f}/{:.4f}, integral time stamps={}, sec/iter={:.4f}'.
          format(batch_idx + 1, len(test_loader), (total_ae/total_sample_num), (loss / n_samples),
                 [model.psi[c].item() for c in range(len(model.psi))],
                 losses[0][0], losses[0][1], losses[1][0], losses[1][1],
                 len(model.Lambda_dict), time_iter / (batch_idx + 1)))

    # Report results for different time slots in the test set
    for c, slot in enumerate(timeslots):
        s = 'Slot {}: '.format(c)
        for event_t in event_types:
            sfx = '' if event_t == event_types[-1] else ', '
            if len(mar[event_t][c]) > 0:
                s += '{} ({} events): MAR={:.2f}+-{:.2f}, HITS_10={:.3f}+-{:.3f}'.\
                    format(event_t, len(mar[event_t][c]), np.mean(mar[event_t][c]), np.std(mar[event_t][c]),
                           np.mean(hits_10[event_t][c]), np.std(hits_10[event_t][c]))
            else:
                s += '{} (no events)'.format(event_t)
            s += sfx
        print(s)

    mar_all, hits_10_all = {}, {}
    for event_t in event_types:
        mar_all[event_t] = []
        hits_10_all[event_t] = []
        for c, slot in enumerate(timeslots):
            mar_all[event_t].extend(mar[event_t][c])
            hits_10_all[event_t].extend(hits_10[event_t][c])

    s = 'All slots: '
    for event_t in event_types:
        sfx = '' if event_t == event_types[-1] else ', '
        if len(mar_all[event_t]) > 0:
            s += '{} ({} events): MAR={:.2f}+-{:.2f}, HITS_10={:.3f}+-{:.3f}'.\
                format(event_t, len(mar_all[event_t]), np.mean(mar_all[event_t]), np.std(mar_all[event_t]),
                       np.mean(hits_10_all[event_t]), np.std(hits_10_all[event_t]))
        else:
            s += '{} (no events)'.format(event_t)
        s += sfx
    print(s)

    return mar_all, hits_10_all, loss / n_samples, total_ae/total_sample_num

def MAR(A_pred, u, v, k, Survival_term):
    '''Computes mean average ranking for a batch of events'''
    ranks = []
    hits_10 = []
    N = len(A_pred)
    Survival_term = jittor.exp(-Survival_term)
    A_pred *= Survival_term
    assert jittor.sum(jittor.isnan(A_pred)) == 0, (jittor.sum(jittor.isnan(A_pred)), Survival_term.min(), Survival_term.max())

    A_pred = A_pred.data.cpu().numpy()

    assert N == len(u) == len(v) == len(k), (N, len(u), len(v), len(k))
    for b in range(N):
        u_it, v_it = u[b].item(), v[b].item()
        assert u_it != v_it, (u_it, v_it, k[b])
        A = A_pred[b].squeeze()
        # remove same node
        idx1 = list(np.argsort(A[u_it])[::-1])
        idx1.remove(u_it)
        idx2 = list(np.argsort(A[v_it])[::-1])
        idx2.remove(v_it)
        rank1 = np.where(np.array(idx1) == v_it) # get nodes most likely connected to u[b] and find out the rank of v[b] among those nodes
        rank2 = np.where(np.array(idx2) == u_it)  # get nodes most likely connected to v[b] and find out the rank of u[b] among those nodes
        assert len(rank1) == len(rank2) == 1, (len(rank1), len(rank2))
        hits_10.append(np.mean([float(rank1[0] <= 9), float(rank2[0] <= 9)]))
        rank = np.mean([rank1[0], rank2[0]])
        assert isinstance(rank, np.float), (rank, rank1, rank2, u_it, v_it, idx1, idx2)
        ranks.append(rank)
    return ranks, hits_10

if __name__ == '__main__':
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description='DyRep Model Training Parameters')

    parser.add_argument('--dataset', type=str, default='github', choices=['github', 'social', 'wikipedia', 'reddit', 'synthetic'])
    parser.add_argument('--data_dir', type=str, default='./Github/')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--hidden_dim', type=int, default=32, help='hidden layer dimension in DyRep')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--lr_decay_step', type=str, default='20', help='number of epochs after which to reduce lr')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--all_comms', type=bool, default=False, help='assume all of the links in Jodie as communication or not')
    parser.add_argument('--include_link_feat', type=bool, default=False, help='include link features or not')
    args = parser.parse_args()
    args.lr_decay_step = list(map(int, args.lr_decay_step.split(','))) # "20,40,60"->[20,40,60]

    # Set seed
    np.random.seed(args.seed)
    rnd = np.random.RandomState(args.seed)

    jittor.set_global_seed(args.seed)
    jittor.flags.lazy_execution=0

    if args.dataset=='github':
        train_set = GithubDataset('train', None)
        test_set = GithubDataset('test', None)
        initial_embeddings = np.random.randn(train_set.N_nodes, args.hidden_dim) #The embedding of each node is randomly generated with a dimension of hidden_dim
        A_initial = train_set.get_Adjacency() #Adjacency matrix, with dimensions N × N


    time_bar_initial = np.zeros((train_set.N_nodes, 1)) + train_set.FIRST_DATE.timestamp()
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    test_reoccur_dict, test_reoccur_time_ts, test_reoccur_time_hr = get_return_time(test_set)
    initial_embeddings = [jittor.randn(train_set.N_nodes, args.hidden_dim)]  # Jittor Var
    InitialEmb.get_instance().initialize(
        num_nodes=train_set.N_nodes,
        hidden_dim=args.hidden_dim,
        device=args.device,
        initial_embs=initial_embeddings
    )
    
    # generate dygraph from A_initial
    dygraph = DiAdjList()
    dygraph.add_vertexes(train_set.N_nodes)
    # Iterate through the adjacency matrix and add edges
    for src in range(train_set.N_nodes):
        for dst in range(train_set.N_nodes):
            if A_initial[src, dst] == 1:  
                edge = TEdge(src=src, dst=dst, edge_data=EdgeData(value=1.0))  
                dygraph.add_edge(edge)  

    def normalize_td(data_set):
        dic = defaultdict(list)
        for src, dst, _, t in data_set.all_events:
            dic[src].append(t.timestamp())
            dic[dst].append(t.timestamp())
        all_diff = []
        all_td = []
        for k, v in dic.items():
            ts = np.array(v)
            td = np.diff(ts)
            all_diff.append(td)
            if len(v) >= 2:
                timestamp = np.array(list(map(lambda x: datetime.fromtimestamp(x), v)))
                delta = np.diff(timestamp)
                all_td.append(delta)
        all_diff = np.concatenate(all_diff)
        all_td = np.concatenate(all_td)
        all_td_hr = np.array(list(map(lambda x: round(x.days * 24 + x.seconds / 3600, 3), all_td)))
        return all_diff.mean(), all_diff.std(), all_diff.max(), \
               round(all_td_hr.mean(), 3), round(all_td_hr.std(), 3), round(all_td_hr.max(), 3)

    train_td_mean, train_td_std, train_td_max, train_td_hr_mean, train_td_hr_std, train_td_hr_max = normalize_td(train_set)
    end_date = test_set.END_DATE

    model = DyRep(num_nodes=train_set.N_nodes,
                  hidden_dim=args.hidden_dim,
                  random_state= rnd,
                  first_date=train_set.FIRST_DATE,
                  end_datetime=end_date,
                  num_neg_samples=10,
                  num_time_samples=5,
                  device=args.device,
                  train_td_max=train_td_max,     
                  all_comms=args.all_comms)
    
    print(train_set.N_nodes)
    print(model)
    print('number of training parameters: %d' %
          np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()]))

    params_main = [param for param in model.parameters() if param.requires_grad]

    optimizer = optim.Adam(params_main, lr=args.lr, betas=(0.5, 0.999)) 
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_step, gamma=0.5)

    for arg in vars(args):
        print(arg, getattr(args, arg))

    dt = datetime.now()
    print('start time:', dt)
    experiment_ID = '%s_%06d' % (platform.node(), dt.microsecond)
    print('experiment_ID: ', experiment_ID)

    epoch_start = 1
    batch_start = 0
    total_losses = []
    total_losses_lambda, total_losses_surv = [], []
    test_MAR, test_HITS10, test_loss = [], [], []
    all_test_mae, all_test_loss = [], []
    all_test_ap, all_test_auc = [], []
    first_batch = []
    for epoch in range(epoch_start, args.epochs + 1):
        model.train()
        if not isinstance(A_initial, list):
            A_initial = [A_initial]
        
        node_degree_initial = []
        for at, A in enumerate(A_initial):
            node_degree_initial.append(np.sum(A, axis=1))
        if len(A_initial) == 1: A_initial = A_initial[0]
        time_bar = copy.deepcopy(time_bar_initial)
        model.reset_state(node_embeddings_initial=initial_embeddings,
                          dygraph=dygraph,
                          node_degree_initial=node_degree_initial,
                          time_bar=time_bar,
                          resetS=(epoch==epoch_start))
        train_loader.dataset.time_bar = time_bar
        test_loader.dataset.time_bar = time_bar

        start = time.time()
        total_loss = 0
        total_loss_lambda, total_loss_surv = 0, 0
        for batch_idx, data_batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            data_batch[2] = data_batch[2].float().to(args.device)
            data_batch[4] = data_batch[4].double().to(args.device)
            data_batch[5] = data_batch[5].double()# no need of GPU
            if args.include_link_feat:
                data_batch[6] = data_batch[6].float().to(args.device)
            
            output = model(data_batch)
            losses = [-jittor.sum(jittor.log(output[0]) + 1e-10), jittor.sum(output[1])]
            loss = jittor.sum(jittor.stack(losses))/args.batch_size
            optimizer.backward(loss)
            optimizer.clip_grad_norm(100, norm_type=2)

            optimizer.step()
            # model.psi.data = jittor.clamp(model.psi.data, 1e-1, 1e+3)  # to prevent overflow in computing Lambda
            # model.psi = jittor.clamp(model.psi, 1e-1, 1e+3)
            time_iter = time.time() - start
            #model.z = model.z.detach()  # to reset the computational graph and avoid backpropagating second time
            model.dz.back_to_initial()
            model.S = model.S.detach()
            if batch_idx % 100 == 0:
                if batch_idx == 0:
                    first_batch.append(loss.item())
                print("Training epoch {}, batch {}/{}, loss {}, loss_lambda {}, loss_surv {}".format(
                    epoch, batch_idx+1, len(train_loader), loss, losses[0]/args.batch_size, losses[1]/args.batch_size))
                # result = test_time_pred(model, test_reoccur_dict, test_reoccur_time_hr)
                # result = test(model, test_reoccur_dict)
            total_loss += loss*args.batch_size
            total_loss_lambda += losses[0]
            total_loss_surv += losses[1]
            scheduler.step()
        
        total_loss = float(total_loss)/len(train_set.all_events)
        total_loss_lambda = float(total_loss_lambda)/len(train_set.all_events)
        total_loss_surv = float(total_loss_surv)/len(train_set.all_events)

        total_losses.append(total_loss)
        total_losses_lambda.append(total_loss_lambda)
        total_losses_surv.append(total_loss_surv)
        print("Training epoch {}/{}, time per batch {}, loss {}, loss_lambda {}, loss_surv {}".format(
            epoch, args.epochs + 1, time_iter/float(batch_idx+1), total_loss, total_loss_lambda, total_loss_surv))

        print("Testing Start")
        test_mae, test_loss, test_ap, test_auc = test_all(model, test_reoccur_dict, test_reoccur_time_hr)
        all_test_mae.append(test_mae)
        all_test_loss.append(test_loss)
        all_test_ap.append(test_ap)
        all_test_auc.append(test_auc)
        print('\nTEST epoch {}/{}, loss={:.3f}, time prediction MAE {}, ap {}, auc{}'.format(
            epoch, args.epochs + 1, test_loss, test_mae, test_ap, test_auc))

        # test_mae, test_loss = test_time_pred(model, test_reoccur_dict, test_reoccur_time_hr)
        # all_test_mae.append(test_mae)
        # all_test_loss.append(test_loss)
        # result = test(model, test_reoccur_dict)
        print("Test end")
    #
    #     # result = test(model, n_test_batches=None)
    #     # test_MAR.append(np.mean(result[0]['Com']))
    #     # test_HITS10.append(np.mean(result[1]['Com']))
    #     # test_loss.append(result[2])
    #     # print("Testing results: MAR {}, HITS10 {}, test_loss {}".format(test_MAR[-1], test_HITS10[-1], test_loss[-1]))
    #
    # print('end time:', datetime.now())


    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, args.epochs + 1), np.array(total_losses), 'k', label='total loss')
    plt.plot(np.arange(1, args.epochs + 1), np.array(total_losses_lambda), 'r', label='loss events')
    plt.plot(np.arange(1, args.epochs + 1), np.array(total_losses_surv), 'b', label='loss nonevents')
    plt.legend()
    plt.title("DyRep, training loss")
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, args.epochs + 1), np.array(first_batch), 'r')
    plt.title("DyRep, loss for the first batch for each epoch")
    fig.savefig('dyrep_train.png')

    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(np.arange(1, args.epochs + 1), np.array(all_test_loss), 'k', label='total loss')
    plt.title("DyRep, test loss")
    plt.subplot(1, 3, 2)
    plt.plot(np.arange(1, args.epochs + 1), np.array(all_test_ap), 'r')
    plt.title("DyRep, test ap")
    plt.subplot(1, 3, 3)
    plt.plot(np.arange(1, args.epochs + 1), np.array(all_test_mae), 'r')
    plt.title("DyRep, test mae")
    fig.savefig('dyrep_test.png')

    end_time = time.perf_counter()  # 结束时间
    elapsed_time = end_time - start_time
    print(f"run time：{elapsed_time:.6f} seconds")