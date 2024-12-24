from typing import List

import jittor as jt
import numpy as np

class TemporalDataLoader:
    def __init__(self, data, batch_size=1, neg_sampling_ratio=None, drop_last=False, num_neg_sample=None, neg_samples=None):
        self.data = data
        self.batch_size = batch_size
        self.neg_sampling_ratio = neg_sampling_ratio
        self.num_neg_sample = num_neg_sample
        self.neg_samples = neg_samples
        self.min_dst = int(data.dst.min())
        self.max_dst = int(data.dst.max())

        data_len = len(data.src)
        self.arange = np.arange(0, data_len, batch_size)
        if drop_last and data_len % batch_size != 0:
            self.arange = self.arange[:-1]

    def __len__(self):
        return len(self.arange)

    def __iter__(self):
        for start in self.arange:
            end = start + self.batch_size
            batch = self.data[start:end]
            n_ids = [batch.src, batch.dst]

            if self.neg_samples is not None:
                neg_dst = self.neg_samples[start:end]
                batch.neg_dst = neg_dst
                n_ids += [batch.neg_dst]
            
            if self.num_neg_sample is not None and self.num_neg_sample > 0:
                neg_dst_size = self.num_neg_sample * len(batch.dst)
                neg_dst = jt.randint(low=self.min_dst, high=self.max_dst + 1, shape=(neg_dst_size,), dtype=batch.dst.dtype)
                batch.neg_dst = neg_dst
                n_ids += [batch.neg_dst]

            if self.neg_sampling_ratio is not None and self.neg_sampling_ratio > 0:
                neg_dst_size = round(self.neg_sampling_ratio * len(batch.dst))
                neg_dst = jt.randint(low=self.min_dst, high=self.max_dst + 1, shape=(neg_dst_size,), dtype=batch.dst.dtype)
                batch.neg_dst = neg_dst
                n_ids += [batch.neg_dst]

            batch.n_id = jt.concat(n_ids).unique()
            yield batch