'''
Description:
Author: zhengyp
Date: 2025-07-13
'''
import os
import os.path as osp
import pdb
import zipfile

import pandas as pd
import numpy as np
import jittor as jt
from jittor_geometric.data import InMemoryDataset, Data, download_url

class DataStruct:
    def __init__(self):
        self._data_dict = {}

    def update_tensor(self, name: str, value: jt.Var):
        # 确保 value 不再参与 autograd（重要）
        # value = value.detach()

        if name not in self._data_dict:
            self._data_dict[name] = value
        else:
            if not isinstance(self._data_dict[name], np.ndarray):
                raise ValueError(f"{name} is not a Jittor tensor.")
            self._data_dict[name] = np.concatenate([self._data_dict[name], value], axis=0)

    def get_tensor(self, name: str):

        return self._data_dict.get(name, None)

class TopkMetric:
    def __init__(self, k=10, metric_decimal_place=4):
        self.decimal_place = metric_decimal_place
        self.topk = [k] if isinstance(k, int) else list(k)

    def used_info(self, rec_mat):
        k = max(self.topk)
        topk_idx, pos_len_list = jt.split(rec_mat, [k, 1], dim=1)
        return topk_idx.bool().numpy(), pos_len_list.squeeze(1).numpy()

    def topk_result(self, metric, value):
        metric_dict = {}
        avg_result = value.mean(dim=0)
        for k in self.topk:
            key = "{}@{}".format(metric, k)
            metric_dict[key] = round(avg_result[k - 1].item(), self.decimal_place)
        return metric_dict

class Hit(TopkMetric):
    def calculate_metric(self, rec_mat):
        pos_index, _ = self.used_info(rec_mat)
        result = self.metric_info(pos_index)
        metric_dict = self.topk_result("hit", result)
        return metric_dict

    def metric_info(self, pos_index):
        result = jt.cumsum(pos_index, dim=1)
        return (result > 0).astype(jt.int)

class MRR(TopkMetric):
    def calculate_metric(self, rec_mat):
        pos_index, _ = self.used_info(rec_mat)
        result = self.metric_info(pos_index)
        metric_dict = self.topk_result("mrr", result)
        return metric_dict

    def metric_info(self, pos_index):
        idxs, _ = jt.argmax(pos_index, dim=1)  # pos_index.argmax(dim=1)
        result = jt.zeros((pos_index.shape), dtype=jt.float)
        for row in range(pos_index.shape[0]):
            idx = int(idxs[row].item())  # 转换为Python标量
            if pos_index[row, idx] > 0:
                # 使用jt.where替代切片赋值
                mask = jt.array([i >= idx for i in range(pos_index.shape[1])], dtype=jt.bool)
                result[row] = jt.where(mask, 1.0 / (idx + 1), 0.0)
        return result

class NDCG(TopkMetric):
    def calculate_metric(self, rec_mat):
        pos_index, pos_len = self.used_info(rec_mat)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result("ndcg", result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        len_rank = jt.full_like(pos_len, pos_index.shape[1])
        idcg_len = jt.minimum(pos_len, len_rank)

        iranks = jt.arange(1, pos_index.shape[1] + 1).unsqueeze(0).broadcast(pos_index.shape)
        idcg = 1.0 / jt.log2(iranks + 1)
        idcg = jt.cumsum(idcg, dim=1)

        for row in range(idcg.shape[0]):
            idx = int(idcg_len[row].item())
            if idx < idcg.shape[1]:
                idcg[row, idx:] = idcg[row, idx - 1:idx]

        ranks = jt.arange(1, pos_index.shape[1] + 1).unsqueeze(0).broadcast(pos_index.shape)
        dcg = 1.0 / jt.log2(ranks + 1)
        dcg = dcg * pos_index
        dcg = jt.cumsum(dcg, dim=1)

        result = dcg / idcg
        return result

class Recall(TopkMetric):
    def calculate_metric(self, rec_mat):
        pos_index, pos_len = self.used_info(rec_mat)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result("recall", result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        return jt.cumsum(pos_index, dim=1) / pos_len.reshape(-1, 1)

def _ensure_dir(path: str):
    """Create the directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def _zero_based(df: pd.DataFrame, cols):
    """
    Convert the specified integer ID columns to zero-based indexing.
    If the minimum value in the column is 1, subtract 1 from all values.
    """
    for c in cols:
        df[c] = df[c].astype(np.int64)
        if df[c].min() == 1:
            df[c] = df[c] - 1
    return df

def _df_to_edge_index(df):
    src = jt.array(df["user_id"].to_numpy(), dtype="int32")
    dst = jt.array(df["item_id"].to_numpy(), dtype="int32")
    return jt.stack([src, dst], dim=0)  # [2, E]

def _df_to_edge_attr(df, cols):
    feats = []
    for c in cols:
        if c in df.columns:
            feats.append(jt.array(df[c].to_numpy().astype(np.float32)))
    return jt.stack(feats, dim=1) if feats else None

def split_dataset(interactions, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, shuffle=True, group_by=None, seed=42):
    """
    Split a dataset into train/validation/test sets by given ratios.
    Optionally split within each group defined by `group_by` before combining results.

    Args:
        interactions (pd.DataFrame): The interaction records.
        train_ratio (float): Proportion of the training set.
        val_ratio (float): Proportion of the validation set.
        test_ratio (float): Proportion of the test set.
        shuffle (bool): Whether to shuffle before splitting.
        seed (int): Random seed for reproducibility.
        group_by (str or None): Optional column name to group by before splitting.

    Returns:
        tuple: (train_df, valid_df, test_df, used_interactions)
            train_df: DataFrame for training set
            valid_df: DataFrame for validation set
            test_df: DataFrame for test set
            used_interactions: DataFrame after any shuffling applied
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    def _split_one(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split one DataFrame according to the ratios."""
        n = len(df)
        n_train = int(train_ratio * n)  # round down
        n_val   = int(val_ratio * n)    # round down
        # The remaining samples go to test
        train = df.iloc[:n_train]
        valid = df.iloc[n_train:n_train + n_val]
        test  = df.iloc[n_train + n_val:]
        return train, valid, test

    if group_by is None:
        # No grouping: shuffle the entire dataset, then split
        used = interactions
        if shuffle:
            used = used.sample(frac=1, random_state=seed).reset_index(drop=True)

        train, valid, test = _split_one(used)
        return train, valid, test, used
    else:
        # Grouping mode: shuffle and split within each group separately
        if group_by not in interactions.columns:
            raise KeyError(f"Column not found for group_by: {group_by}")

        rng = np.random.RandomState(seed)

        train_parts, val_parts, test_parts, used_parts = [], [], [], []
        # Keep group order as in original DataFrame (sort=False)
        for _, gdf in interactions.groupby(group_by, sort=False):
            g_used = gdf
            if shuffle:
                # Use a different but reproducible sub-seed for each group
                g_seed = int(rng.randint(0, 2**31 - 1))
                g_used = g_used.sample(frac=1, random_state=g_seed).reset_index(drop=True)

            tr, va, te = _split_one(g_used)
            train_parts.append(tr)
            val_parts.append(va)
            test_parts.append(te)
            used_parts.append(g_used)

        # Concatenate splits from all groups
        train = pd.concat(train_parts, ignore_index=True)
        valid = pd.concat(val_parts, ignore_index=True)
        test  = pd.concat(test_parts,  ignore_index=True)
        used  = pd.concat(used_parts,  ignore_index=True)

        pdb.set_trace()
        return train, valid, test, used

class RecSysBase(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None,
                 train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42,
                 edge_attr_columns=None,
                 group_by='user_id',
                 shuffle=True, with_aux: bool = False):
        self.name = name
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio
        self._seed = seed
        self._edge_attr_columns = edge_attr_columns
        self.group_by = group_by
        self._shuffle = shuffle
        self.with_aux = with_aux
        super().__init__(root, transform, pre_transform)

        proc_path = self.processed_paths[0]
        if not osp.exists(proc_path):
            self.process()
        self.data, self.slices = jt.load(proc_path)

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.pkl"

    @property
    def raw_file_names(self):
        raise NotImplementedError

    def read_raw(self):
        """
        Must return either:
          - interactions_df                        (if self.with_aux == False)
          - (interactions_df, {'items': df?, 'users': df?})  (if self.with_aux == True)
        """
        raise NotImplementedError

    def process(self):
        _ensure_dir(self.processed_dir)
        raw = self.read_raw()
        if self.with_aux:
            interactions, aux = raw
            items = aux.get('items', None)
            users = aux.get('users', None)
        else:
            interactions = raw
            items = None
            users = None

        # Zero-based IDs
        interactions = _zero_based(interactions, ["user_id", "item_id"])
        num_users = int(interactions["user_id"].max()) + 1
        num_items = int(interactions["item_id"].max()) + 1
        print(f"Number of users: {num_users}")
        print(f"Number of items: {num_items}")
        print(f"Number of interactions: {len(interactions)}")

        train_df, val_df, test_df, inter_used = split_dataset(
            interactions,
            train_ratio=self._train_ratio,
            val_ratio=self._val_ratio,
            test_ratio=self._test_ratio,
            shuffle=self._shuffle, group_by=self.group_by,
            seed=self._seed
        )
        self.train_df = train_df
        train_edge_index = _df_to_edge_index(train_df)
        val_edge_index = _df_to_edge_index(val_df)
        test_edge_index = _df_to_edge_index(test_df)
        edge_index_all = _df_to_edge_index(inter_used)

        if self._edge_attr_columns is not None:
            train_edge_attr = _df_to_edge_attr(train_df, self._edge_attr_columns)
            val_edge_attr = _df_to_edge_attr(val_df, self._edge_attr_columns)
            test_edge_attr = _df_to_edge_attr(test_df, self._edge_attr_columns)
            edge_attr_all = _df_to_edge_attr(inter_used, self._edge_attr_columns)

        E = len(inter_used)
        n_train, n_val = len(train_df), len(val_df)
        print(f'Edges for train: {n_train}, valid: {n_val}, test: {E - n_train - n_val}')

        data = Data()
        data.edge_index = edge_index_all
        if self._edge_attr_columns is not None:
            data.edge_attr = edge_attr_all
            data.train_edge_attr = train_edge_attr
            data.val_edge_attr = val_edge_attr
            data.test_edge_attr = test_edge_attr
        data.num_users = num_users
        data.num_items = num_items
        data.num_nodes = num_users + num_items
        data.train_edge_index = train_edge_index
        data.val_edge_index = val_edge_index
        data.test_edge_index = test_edge_index

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        jt.save(self.collate([data]), self.processed_paths[0])

        if self.with_aux:
            if items is not None:
                items_path = osp.join(self.processed_dir, "items.csv")
                items.to_csv(items_path, index=False)
                self.items_df = items  # 运行期可用
            if users is not None:
                users_path = osp.join(self.processed_dir, "users.csv")
                users.to_csv(users_path, index=False)
                self.users_df = users

class MovieLens1M(RecSysBase):
    """
    MovieLens-1M dataset with auto-download from Recbole:
      https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/ml-1m.zip

    Expected (after extraction) in raw_dir:
      - ml-1m.item
      - ml-1m.user
      - ml-1m.inter

    Files are tab-separated; first header row is skipped (skiprows=1).
    """
    url = "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/ml-1m.zip"

    def __init__(self, root, transform=None, pre_transform=None,
                 train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42,
                 shuffle=True, with_aux: bool=False):
        super().__init__(root=root, name="ml-1m",
                         transform=transform, pre_transform=pre_transform,
                         train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                         seed=seed,
                         edge_attr_columns=None,
                         shuffle=shuffle, with_aux=with_aux)
        print('MovieLens1M - with_aux:', self.with_aux)

    @property
    def raw_file_names(self):
        return ["ml-1m.item", "ml-1m.user", "ml-1m.inter"]

    def _raw_exists(self):
        return all(osp.exists(osp.join(self.raw_dir, f)) for f in self.raw_file_names)

    def download(self):
        """Download and extract ml-1m.zip into raw_dir (idempotent)."""
        if self._raw_exists():
            return

        os.makedirs(self.raw_dir, exist_ok=True)
        zip_path = osp.join(self.raw_dir, "ml-1m.zip")

        # Download the zip
        download_url(self.url, self.raw_dir)

        # Extract
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.raw_dir)

        candidates = [
            self.raw_dir,
            osp.join(self.raw_dir, "ml-1m"),
            osp.join(self.raw_dir, "MovieLens-1M"),
        ]
        # Try to locate the three files and move them to raw_dir if needed
        for base in candidates:
            item_src = osp.join(base, "ml-1m.item")
            user_src = osp.join(base, "ml-1m.user")
            inter_src = osp.join(base, "ml-1m.inter")
            if all(osp.exists(p) for p in [item_src, user_src, inter_src]):
                # If base is not raw_dir, move files up
                if base != self.raw_dir:
                    for src in [item_src, user_src, inter_src]:
                        dst = osp.join(self.raw_dir, osp.basename(src))
                        if not osp.exists(dst):
                            os.replace(src, dst)
                break

        assert self._raw_exists(), (
            "Failed to locate ml-1m.item/ml-1m.user/ml-1m.inter after extraction."
        )

    def read_raw(self):
        inter_file = osp.join(self.raw_dir, "ml-1m.inter")

        interactions = pd.read_csv(inter_file, sep='\t', engine='python', skiprows=1,
                                   names=['user_id', 'item_id', 'rating', 'timestamp'])

        interactions['user_id'] = interactions['user_id'].astype(int)
        interactions['item_id'] = interactions['item_id'].astype(int)
        if not self.with_aux:
            return interactions

        items = None
        users = None

        print('load properties')
        item_file = osp.join(self.raw_dir, "ml-1m.item")
        if osp.exists(item_file):
            items = pd.read_csv(
                item_file, sep='\t', engine='python', skiprows=1,
                names=['item_id', 'movie_title', 'release_year', 'genre'],
                usecols=[0, 1, 2, 3]
            )
            items["item_id"] = items["item_id"].astype(int)

        user_file = osp.join(self.raw_dir, "ml-1m.user")
        if osp.exists(user_file):
            users = pd.read_csv(
                user_file, sep='\t', engine='python', skiprows=1, header=None,
                names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
            )
            users["user_id"] = users["user_id"].astype(int)

        return interactions, {"items": items, "users": users}

class MovieLens100K(RecSysBase):
    """MovieLens-100K (RecBole processed).

    Downloads: https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/ml-100k.zip
    
    Expected in raw_dir after extraction:
    
    - ml-100k.item
    - ml-100k.user  
    - ml-100k.inter
    """
    url = "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/ml-100k.zip"

    def __init__(self, root, transform=None, pre_transform=None,
                 train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42,
                 shuffle=True, with_aux: bool=False):
        super().__init__(root=root, name="ml-100k",
                         transform=transform, pre_transform=pre_transform,
                         train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                         seed=seed, shuffle=shuffle, with_aux=with_aux,
                         edge_attr_columns=("rating", "timestamp"))

    @property
    def raw_file_names(self):
        return ["ml-100k.item", "ml-100k.user", "ml-100k.inter"]

    def _raw_exists(self):
        return all(osp.exists(osp.join(self.raw_dir, f)) for f in self.raw_file_names)

    def download(self):
        if self._raw_exists():
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        zip_path = osp.join(self.raw_dir, "ml-100k.zip")

        # fetch
        download_url(self.url, self.raw_dir)  # -> raw/ml-100k.zip

        # extract
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.raw_dir)

        # normalize location if nested
        candidates = [
            self.raw_dir,
            osp.join(self.raw_dir, "ml-100k"),
            osp.join(self.raw_dir, "MovieLens-100K"),
        ]
        for base in candidates:
            paths = [osp.join(base, n) for n in self.raw_file_names]
            if all(osp.exists(p) for p in paths):
                if base != self.raw_dir:
                    for src in paths:
                        dst = osp.join(self.raw_dir, osp.basename(src))
                        if not osp.exists(dst):
                            os.replace(src, dst)
                break

        assert self._raw_exists(), "ml-100k.* not found after extraction."

    def read_raw(self):
        inter_path = osp.join(self.raw_dir, "ml-100k.inter")
        if not osp.exists(inter_path):
            raise FileNotFoundError(f"Missing: {inter_path}")

        interactions = pd.read_csv(
            inter_path, sep='\t', engine='python', skiprows=1,
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )
        interactions['user_id'] = interactions['user_id'].astype(int)
        interactions['item_id'] = interactions['item_id'].astype(int)

        if not self.with_aux:
            return interactions

        items, users = None, None

        item_path = osp.join(self.raw_dir, "ml-100k.item")
        if osp.exists(item_path):
            items = pd.read_csv(
                item_path, sep='\t', engine='python', skiprows=1,
                names=['item_id', 'movie_title', 'release_year', 'genre'],
                usecols=[0, 1, 2, 3]
            )
            items['item_id'] = items['item_id'].astype(int)

        user_path = osp.join(self.raw_dir, "ml-100k.user")
        if osp.exists(user_path):
            users = pd.read_csv(
                user_path, sep='\t', engine='python', skiprows=1, header=None,
                names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
            )
            users['user_id'] = users['user_id'].astype(int)

        return interactions, {"items": items, "users": users}

class Yelp2018(RecSysBase):
    """Yelp-2018 (RecBole processed).

    Downloads: https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Yelp/yelp2018.zip
    
    Accepts either file naming variant inside the zip:
    
    - yelp2018.item/.user/.inter  (common)
    - yelp-2018.item/.user/.inter (also supported)
    
    After extraction, we normalize to yelp-2018.* in raw_dir.
    """
    url = "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Yelp/yelp2018.zip"

    @property
    def raw_file_names(self):
        return ["yelp-2018.item", "yelp-2018.user", "yelp-2018.inter"]

    def __init__(self, root, transform=None, pre_transform=None,
                 train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42,
                 shuffle=True, with_aux: bool=False):
        super().__init__(root=root, name="yelp-2018",
                         transform=transform, pre_transform=pre_transform,
                         train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                         seed=seed, shuffle=shuffle, with_aux=with_aux,
                         edge_attr_columns=("rating", "timestamp"))

    def _raw_exists(self):
        return all(osp.exists(osp.join(self.raw_dir, f)) for f in self.raw_file_names)

    def download(self):
        if self._raw_exists():
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        zip_path = osp.join(self.raw_dir, "yelp2018.zip")

        download_url(self.url, self.raw_dir)  # -> raw/yelp2018.zip
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.raw_dir)

        # Possible inner names
        variants = [
            ("yelp2018.item", "yelp-2018.item"),
            ("yelp2018.user", "yelp-2018.user"),
            ("yelp2018.inter", "yelp-2018.inter"),
            ("yelp-2018.item", "yelp-2018.item"),
            ("yelp-2018.user", "yelp-2018.user"),
            ("yelp-2018.inter", "yelp-2018.inter"),
        ]
        # search bases
        candidates = [
            self.raw_dir,
            osp.join(self.raw_dir, "yelp2018"),
            osp.join(self.raw_dir, "yelp-2018"),
            osp.join(self.raw_dir, "Yelp-2018"),
        ]

        found = {}
        for base in candidates:
            for src_name, target_name in variants:
                src = osp.join(base, src_name)
                if osp.exists(src):
                    dst = osp.join(self.raw_dir, target_name)
                    if not osp.exists(dst):
                        os.replace(src, dst)
                    found[target_name] = True

        assert self._raw_exists(), "yelp-2018.* not found after extraction."

    def read_raw(self):
        inter_path = osp.join(self.raw_dir, "yelp-2018.inter")
        if not osp.exists(inter_path):
            raise FileNotFoundError(f"Missing: {inter_path}")

        interactions = pd.read_csv(
            inter_path, sep='\t', engine='python', skiprows=1,
            names=['user_id', 'item_id', 'rating', 'timestamp',
                   'useful', 'funny', 'cool', 'review_id']
        )

        if not self.with_aux:
            interactions['user_id'] = interactions['user_id'].astype('category').cat.codes.astype(np.int64)
            interactions['item_id'] = interactions['item_id'].astype('category').cat.codes.astype(np.int64)
            return interactions

        items, users = None, None

        item_path = osp.join(self.raw_dir, "yelp-2018.item")
        if osp.exists(item_path):
            items = pd.read_csv(
                item_path, sep='\t', engine='python', skiprows=1,
                names=['item_id', 'item_name', 'address', 'city','state', 'postal_code',
                       'latitude', 'longitude', 'item_stars', 'item_review_count',
                       'is_open', 'categories']
            )

        user_path = osp.join(self.raw_dir, "yelp-2018.user")
        if osp.exists(user_path):
            users = pd.read_csv(
                user_path, sep='\t', engine='python', skiprows=1,
                names=['user_id', 'user_name', 'user_review_count', 'yelping_since',
                       'user_useful', 'user_funny', 'user_cool', 'elite', 'fans',
                       'average_stars', 'compliment_hot', 'compliment_more',
                       'compliment_profile', 'compliment_cute', 'compliment_list',
                       'compliment_note', 'compliment_plain', 'compliment_cool',
                       'compliment_funny', 'compliment_writer', 'compliment_photos']
            )

        # If both aux frames exist, remap using their unique IDs for consistency
        if items is not None and users is not None and not items.empty and not users.empty:
            item_cats = items['item_id'].astype('category').cat.categories
            user_cats = users['user_id'].astype('category').cat.categories
            item_to_id = {cat: idx for idx, cat in enumerate(item_cats)}
            user_to_id = {cat: idx for idx, cat in enumerate(user_cats)}

            interactions['item_id'] = interactions['item_id'].map(item_to_id)
            interactions['user_id'] = interactions['user_id'].map(user_to_id)
            interactions = interactions.dropna(subset=['user_id', 'item_id'])
            interactions['user_id'] = interactions['user_id'].astype(np.int64)
            interactions['item_id'] = interactions['item_id'].astype(np.int64)

            items = items.copy()
            users = users.copy()
            items['item_id'] = items['item_id'].map(item_to_id).astype(np.int64)
            users['user_id'] = users['user_id'].map(user_to_id).astype(np.int64)
        else:
            # fallback: interactions-only remap
            interactions['user_id'] = interactions['user_id'].astype('category').cat.codes.astype(np.int64)
            interactions['item_id'] = interactions['item_id'].astype('category').cat.codes.astype(np.int64)

        return interactions, {"items": items, "users": users}

if __name__ == '__main__':
    # interactions only
    ds = MovieLens1M(root="./data", with_aux=False)
    data = ds.get(0)

    # with auxiliary metaframes
    # ds_aux = MovieLens1M(root="./data", with_aux=True)
    # data_aux = ds_aux.get(0)

    # ds_aux = MovieLens100K(root="./data", with_aux=True)
    # data_aux = ds_aux.get(0)
    # pdb.set_trace()
    # ds_aux = Yelp2018(root="./data", with_aux=True)
    # data_aux = ds_aux.get(0)
    # pdb.set_trace()
