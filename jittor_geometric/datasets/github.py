import numpy as np
import datetime
import os
import jittor
import pickle
from jittor import dataset
from jittor.dataset import Dataset
from datetime import datetime, timezone, timedelta

class GithubDataset(Dataset):
    def __init__(self, split, data_dir= None):
        super(GithubDataset, self).__init__()
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(__file__),  
                'Github'  
            )
        
        self.TZ = None
        if split == 'train':
            time_start = 0
            time_end = datetime(2013, 8, 31, tzinfo=self.TZ).toordinal()
        elif split == 'test':
            time_start = datetime(2013, 9, 1, tzinfo=self.TZ).toordinal()
            time_end = datetime(2014, 1, 1, tzinfo=self.TZ).toordinal()
        else:
            raise ValueError('invalid split', split)

        self.link_feat = False

        self.FIRST_DATE = datetime(2012, 12, 28, tzinfo=self.TZ)

        self.TEST_TIMESLOTS = [datetime(2013, 9, 1, tzinfo=self.TZ),
                               datetime(2013, 9, 25, tzinfo=self.TZ),
                               datetime(2013, 10, 20, tzinfo=self.TZ),
                               datetime(2013, 11, 15, tzinfo=self.TZ),
                               datetime(2013, 12, 10, tzinfo=self.TZ),
                               datetime(2014, 1, 1, tzinfo=self.TZ)]
        self.END_DATE = self.TEST_TIMESLOTS[-1]


        with open(os.path.join(data_dir, 'github_284users_events_2013.pkl'), 'rb') as f:
            users_events, event_types = pickle.load(f)

        with open(os.path.join(data_dir, 'github_284users_follow_2011_2012.pkl'), 'rb') as f:
            users_follow = pickle.load(f)

        print(event_types)

        self.events2name = {}
        for e in event_types:
            self.events2name[event_types[e]] = e
        print(self.events2name)

        #communication event type
        self.event_types = ['ForkEvent', 'PushEvent', 'WatchEvent', 'IssuesEvent', 'IssueCommentEvent',
                           'PullRequestEvent', 'CommitCommentEvent']
        #association event type
        self.assoc_types = ['FollowEvent']
        self.is_comm = lambda d: self.events2name[d['type']] in self.event_types
        self.is_assoc = lambda d: self.events2name[d['type']] in self.assoc_types

        user_ids = {}
        for id, user in enumerate(sorted(users_events)):
            user_ids[user] = id

        self.N_nodes = len(user_ids)
        
        self.A_initial = np.zeros((self.N_nodes, self.N_nodes))
        for user in users_follow:
            for e in users_follow[user]:
                assert e['type'] in self.assoc_types, e['type']
                if e['login'] in users_events:
                    self.A_initial[user_ids[user], user_ids[e['login']]] = 1

        self.A_last = np.zeros((self.N_nodes, self.N_nodes))
        for user in users_events:
            for e in users_events[user]:
                if self.events2name[e['type']] in self.assoc_types:
                    self.A_last[user_ids[user], user_ids[e['login']]] = 1

        print('\nA_initial', np.sum(self.A_initial))
        print('A_last', np.sum(self.A_last), '\n')

        all_events = []
        for user in users_events:
            if user not in user_ids:
                continue
            user_id = user_ids[user]
            for ind, event in enumerate(users_events[user]):
                event['created_at'] = datetime.fromtimestamp(event['created_at'])
                if event['created_at'].toordinal() >= time_start and event['created_at'].toordinal() <= time_end:
                    if 'owner' in event:
                        if event['owner'] not in user_ids:
                            continue
                        user_id2 = user_ids[event['owner']]
                    elif 'login' in event:
                        if event['login'] not in user_ids:
                            continue
                        user_id2 = user_ids[event['login']]
                    else:
                        raise ValueError('invalid event', event)
                    if user_id != user_id2:
                        all_events.append((user_id, user_id2,
                                           self.events2name[event['type']], event['created_at']))

        self.all_events = sorted(all_events, key=lambda t: t[3].timestamp())
        print('\n%s' % split.upper())
        print('%d events between %d users loaded' % (len(self.all_events), self.N_nodes))
        print('%d communication events' % (len([t for t in self.all_events if t[2] == 1])))
        print('%d assocition events' % (len([t for t in self.all_events if t[2] == 0])))

        self.event_types_num = {self.assoc_types[0]: 0}
        k = 1  # k >= 1 for communication events
        for t in self.event_types:
            self.event_types_num[t] = k
            k += 1

        self.n_events = len(self.all_events)


    def get_Adjacency(self, multirelations=False):
        if multirelations:
            print('warning: Github has only one relation type (FollowEvent), so multirelations are ignored')
        return self.A_initial

    def __len__(self):
        return self.n_events

    def __getitem__(self, index):
        tpl = self.all_events[index]
        if self.link_feat:
            u, v, rel, time_cur, link_feature = tpl
        else:
            u, v, rel, time_cur = tpl
        
        time_delta_uv = np.zeros((2, 4))
        time_bar = self.time_bar.copy()

        for c, j in enumerate([u, v]):
            node_idx = int(j)
            
            last_time = float(self.time_bar[node_idx].item())  
            current_time = float(time_cur.timestamp())  
            
            delta_seconds = current_time - last_time
            
            td = timedelta(seconds=delta_seconds)
            
            time_delta_uv[c] = [
                td.days,
                td.seconds // 3600,
                (td.seconds // 60) % 60,
                td.seconds % 60
            ]
            
            self.time_bar[node_idx] = current_time

        k = self.event_types_num[rel]
        time_cur_ts = np.float64(time_cur.timestamp())
        time_bar = time_bar.astype(np.float64)
        
        if self.link_feat:
            return u, v, time_delta_uv, k, time_bar, time_cur_ts, link_feature
        else:
            return u, v, time_delta_uv, k, time_bar, time_cur_ts