# loader_rahgrec.py

import os
import random
import collections

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from loader_base import DataLoaderBase


class DataLoaderRAHGREC(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        # Load RAW KG (no inverse edges, no relation shifting)
        if self.knowledgegraph:
            kg_data_raw = self.load_kg(self.kg_file)
            # Optional debug dump
            # kg_data_raw.to_csv('kg_data.csv', index=False)

            self.h_list = torch.LongTensor(kg_data_raw['h'].values)
            self.r_list = torch.LongTensor(kg_data_raw['r'].values)
            self.t_list = torch.LongTensor(kg_data_raw['t'].values)
        else:
            kg_data_raw = self.load_kg(self.kg_empty)

        # Optional: constraint maps from RAW KG (stable relation IDs)
        if int(getattr(args, "enable_constraints", 0)) == 1:
            self.constraint_maps = self.build_constraint_maps_from_raw_kg(kg_data_raw)
        else:
            self.constraint_maps = None

        # Build transformed KG (adds inverse edges + shifts relation ids + injects CF edges)
        self.construct_data(kg_data_raw)

        # Role-aware hyperedges must be built AFTER construct_data() because self.n_entities is finalized there.
        if int(getattr(args, "enable_role_hg", 0)) == 1:
            self.role_hyper = self.build_role_hyper_from_raw_kg(kg_data_raw)
        else:
            self.role_hyper = None

        self.print_info(logging)

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()

        # Optional debug; WARNING: dense dump can be huge
        # self.check_kg_sparsity()

        self.incidence_matrix = self.create_incidence_matrix()

    # -------------------------
    # Utilities for sparse ops
    # -------------------------
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data.astype(np.float32))
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def get_D_inv(self, H):
        """
        H: scipy sparse matrix (n_nodes x n_hyperedges)
        returns Dv^-1 and De^-1 (both scipy sparse diagonal matrices)
        """
        dv = np.array(H.sum(axis=1)).flatten()
        de = np.array(H.sum(axis=0)).flatten()

        dv_inv = np.where(dv != 0, 1.0 / dv, 0.0).astype(np.float32)
        de_inv = np.where(de != 0, 1.0 / de, 0.0).astype(np.float32)

        Dv_inv = sp.diags(dv_inv)
        De_inv = sp.diags(de_inv)
        return Dv_inv, De_inv

    # ---------------------------------------------------------
    # Role-aware hypergraph construction (from RAW KG)
    # ---------------------------------------------------------
    def build_role_hyper_from_raw_kg(self, kg_data_raw: pd.DataFrame):
        """
        Build role-specific hyperedge incidence matrices H_role from RAW KG.

        Returns dict:
            role -> {
                'H'      : torch_sparse (n_nodes x n_hyperedges),
                'H_T'    : torch_sparse (n_hyperedges x n_nodes),
                'norm1'  : torch_sparse (n_nodes x n_hyperedges)  = Dv^-1 * H * De^-1,
                'norm2'  : torch_sparse (n_hyperedges x n_nodes)  = H^T,
                'n_hyperedges': int
            }

        Semantics:
        - For each role, relation id = role_rel_map[role]  (RAW relations.txt ids)
        - One hyperedge per unique head node h under that relation.
        - Hyperedge connects h and all its tails t (and includes h itself).
        """
        roles = eval(getattr(self.args, "roles", "[]"))
        role_rel_map = eval(getattr(self.args, "role_rel_map", "{}"))

        if not isinstance(roles, (list, tuple)) or len(roles) == 0:
            raise ValueError(
                "enable_role_hg=1 but args.roles is empty. Provide e.g. --roles \"['dep','topic']\""
            )
        if not isinstance(role_rel_map, dict) or len(role_rel_map) == 0:
            raise ValueError(
                "enable_role_hg=1 but args.role_rel_map is empty. Provide e.g. --role_rel_map \"{'dep':1,'topic':0}\""
            )

        # IMPORTANT: must match embedding size (self.n_entities)
        n_nodes = self.n_entities

        # Sanity: raw KG ids should be within [0, self.n_entities-1]
        raw_max = int(max(kg_data_raw["h"].max(), kg_data_raw["t"].max()))
        if raw_max >= n_nodes:
            raise ValueError(
                f"[RAHGRec] RAW KG has node id {raw_max} but n_entities={n_nodes}. "
                f"Fix entity indexing or ensure construct_data() covers all raw nodes."
            )

        role_hyper = {}

        for role in roles:
            if role not in role_rel_map:
                raise ValueError(
                    f"Role '{role}' missing in role_rel_map. Provided keys: {list(role_rel_map.keys())}"
                )

            rel_id = int(role_rel_map[role])
            sub = kg_data_raw[kg_data_raw["r"] == rel_id][["h", "t"]].values

            if len(sub) == 0:
                print(f"[RAHGRec] role={role} rel_id={rel_id} -> 0 triples (check role_rel_map vs relations.txt)")
                role_hyper[role] = {
                    "H": None, "H_T": None, "norm1": None, "norm2": None, "n_hyperedges": 0
                }
                continue

            heads = sub[:, 0].astype(int)
            tails = sub[:, 1].astype(int)

            unique_heads, head_to_eid = np.unique(heads, return_inverse=True)
            n_hyperedges = int(unique_heads.shape[0])

            row_idx, col_idx = [], []
            for (h, t, eid) in zip(heads, tails, head_to_eid):
                # tail membership
                row_idx.append(int(t))
                col_idx.append(int(eid))
                # head membership (owner)
                row_idx.append(int(h))
                col_idx.append(int(eid))

            data = np.ones(len(row_idx), dtype=np.float32)
            H = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n_nodes, n_hyperedges)).tocsr()

            # Optional sparsification
            role_alpha = float(getattr(self.args, "role_alpha", 0.0))
            if role_alpha > 0 and H.shape[1] > 0:
                de = np.array(H.sum(axis=0)).flatten()
                thr = role_alpha * (float(de.mean()) if de.size > 0 else 0.0)
                keep = np.where(de >= thr)[0]
                H = H[:, keep].tocsr()
                n_hyperedges = int(H.shape[1])

            # Normalization
            Dv_inv, De_inv = self.get_D_inv(H)
            norm1 = Dv_inv.dot(H).dot(De_inv)   # (n_nodes x n_hyperedges)
            H_T = H.transpose().tocsr()         # (n_hyperedges x n_nodes)

            print(f"[RAHGRec] role={role} rel_id={rel_id} triples={len(sub)} hyperedges={n_hyperedges} n_nodes={n_nodes}")

            role_hyper[role] = {
                "H": self.sparse_mx_to_torch_sparse_tensor(H.tocoo()),
                "H_T": self.sparse_mx_to_torch_sparse_tensor(H_T.tocoo()),
                "norm1": self.sparse_mx_to_torch_sparse_tensor(norm1.tocoo()),
                "norm2": self.sparse_mx_to_torch_sparse_tensor(H_T.tocoo()),
                "n_hyperedges": int(n_hyperedges),
            }

        return role_hyper

    # ---------------------------------------------------------
    # Constraint maps from RAW KG (optional)
    # ---------------------------------------------------------
    def build_constraint_maps_from_raw_kg(self, kg_data_raw: pd.DataFrame):
        dep_r = getattr(self.args, "rel_dep", -1)
        lic_r = getattr(self.args, "rel_license", -1)
        vuln_r = getattr(self.args, "rel_vuln", -1)

        dep_map = collections.defaultdict(set)  # lib -> set(dependencies)
        license_map = {}                       # lib -> license node id
        vuln_set = set()                       # libraries flagged vulnerable

        if dep_r != -1:
            sub = kg_data_raw[kg_data_raw["r"] == dep_r][["h", "t"]].values
            for h, t in sub:
                dep_map[int(h)].add(int(t))

        if lic_r != -1:
            sub = kg_data_raw[kg_data_raw["r"] == lic_r][["h", "t"]].values
            for h, t in sub:
                license_map[int(h)] = int(t)

        if vuln_r != -1:
            sub = kg_data_raw[kg_data_raw["r"] == vuln_r][["h", "t"]].values
            for h, _ in sub:
                vuln_set.add(int(h))

        license_blocklist = set(eval(getattr(self.args, "license_blocklist", "[]")))
        block_vulnerable = int(getattr(self.args, "block_vulnerable", 1))

        return {
            "dep_map": dep_map,
            "license_map": license_map,
            "vuln_set": vuln_set,
            "license_blocklist": license_blocklist,
            "block_vulnerable": block_vulnerable,
        }

    # ---------------------------------------------------------
    # Original PYREC hypergraph incidence matrix (CF hyperedges)
    # ---------------------------------------------------------
    def create_incidence_matrix(self):
        n_entities = self.n_entities
        n_hyperedges = len(self.cf_train_data[0])

        row_indices = []
        col_indices = []

        for hyperedge_idx, (project_id, library_id) in enumerate(zip(self.cf_train_data[0], self.cf_train_data[1])):
            row_indices.append(project_id)
            row_indices.append(library_id)
            col_indices.append(hyperedge_idx)
            col_indices.append(hyperedge_idx)

        data = [1] * len(row_indices)
        incidence_matrix = sp.coo_matrix((data, (row_indices, col_indices)), shape=(n_entities, n_hyperedges))
        print(f"[RAHGRec] Created Incidence Matrix with shape {incidence_matrix.shape} and non-zero elements {incidence_matrix.nnz}")
        return incidence_matrix

    # ---------------------------------------------------------
    # Debug: KG sparsity (optional)
    # ---------------------------------------------------------
    def check_kg_sparsity(self):
        if not hasattr(self, 'A_in'):
            raise AttributeError("Adjacency matrix is not constructed yet.")

        if isinstance(self.A_in, torch.sparse.FloatTensor) or self.A_in.is_sparse:
            total_elements = self.A_in.shape[0] * self.A_in.shape[1]
            non_zero_elements = self.A_in._nnz()
        else:
            raise TypeError("A_in is not a sparse tensor.")

        sparsity = 1 - (non_zero_elements / total_elements)
        density = non_zero_elements / total_elements

        print(f"Total elements: {total_elements}")
        print(f"Non-zero elements: {non_zero_elements}")
        print(f"Sparsity: {sparsity}")
        print(f"Density: {density}")

        # WARNING: can be huge
        # adj_matrix = self.A_in.to_dense().cpu().numpy()
        # pd.DataFrame(adj_matrix).to_csv('adj_matrix_rahgrec.csv', index=False)

    # ---------------------------------------------------------
    # Construct transformed KG
    # ---------------------------------------------------------
    def construct_data(self, kg_data):
        kg_data = kg_data.dropna(subset=['h', 'r', 't'])

        if 'r' not in kg_data.columns:
            raise ValueError("The 'r' column is missing.")

        kg_data['h'] = pd.to_numeric(kg_data['h'], errors='coerce')
        kg_data['r'] = pd.to_numeric(kg_data['r'], errors='coerce')
        kg_data['t'] = pd.to_numeric(kg_data['t'], errors='coerce')
        kg_data = kg_data.dropna()

        if kg_data.empty:
            raise ValueError("KG is empty after filtering invalid values.")

        kg_data = kg_data.astype({'h': int, 'r': int, 't': int})

        # Add inverse edges
        n_relations = int(kg_data['r'].max()) + 1
        inverse_kg_data = kg_data.copy().rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations

        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        # Shift relations by +2 to reserve r=0/1 for CF edges
        kg_data['r'] += 2
        self.n_relations = int(kg_data['r'].max()) + 1
        print("[RAHGRec] Number of relations (incl. inverse + shifted) ->", self.n_relations)

        self.n_entities = int(max(kg_data['h'].max(), kg_data['t'].max())) + 1
        print("[RAHGRec] Number of entities ->", self.n_entities)

        self.n_users_entities = self.n_entities

        # Inject CF edges as r=0 and r=1 (reverse)
        cf2kg_train_data = pd.DataFrame(
            np.zeros((self.n_cf_train, 3), dtype=np.int32),
            columns=['h', 'r', 't']
        )
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]

        inverse_cf2kg_train_data = pd.DataFrame(
            np.ones((self.n_cf_train, 3), dtype=np.int32),
            columns=['h', 'r', 't']
        )
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        h_list, t_list, r_list = [], [], []

        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for _, row in self.kg_train_data.iterrows():
            h, r, t = int(row['h']), int(row['r']), int(row['t'])
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)

    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        return torch.sparse.FloatTensor(i, v, torch.Size(coo.shape))

    # ---------------------------------------------------------
    # Laplacian/Adjacency creation
    # ---------------------------------------------------------
    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
            self.adjacency_dict[r] = adj

    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1, dtype=np.float32))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1, dtype=np.float32))
            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            print("[RAHGRec] - Laplacian r ->", r)
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())

    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------
    def print_info(self, logging):
        logging.info(' ---------- summary -----------')
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)
