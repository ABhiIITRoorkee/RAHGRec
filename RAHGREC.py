# RAHGREC.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.0)


class HyperAggregator(nn.Module):
    """
    One message passing layer that can combine:
      (1) KG adjacency propagation (A_in)
      (2) Role-aware hypergraph propagation (role_hyper: dict role -> {norm1, norm2})
    """

    def __init__(self, in_dim, out_dim, dropout, aggregator_type, attention):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type
        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()
        self.attention = attention

        if self.aggregator_type == "bi-interaction":
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
        else:
            raise NotImplementedError(f"Unknown aggregator_type={self.aggregator_type}")

    @staticmethod
    def _spmm(mat, x):
        """Sparse/dense matmul mat @ x. mat can be torch sparse or dense."""
        if mat is None:
            return None
        if getattr(mat, "is_sparse", False):
            return torch.sparse.mm(mat, x)
        return torch.matmul(mat, x)

    def forward(
        self,
        ego_embeddings,
        A_in=None,
        role_hyper=None,
        role_order=None,
        role_weights=None,
        use_kg=True,
        use_role_hg=False,
        dbg_state=None,   # optional dict for one-time debug prints
    ):
        """
        ego_embeddings: (n_entities, in_dim)
        A_in:           (n_entities, n_entities) torch.sparse
        role_hyper: dict role -> {'norm1': (n_nodes x n_hyperedges), 'norm2': (n_hyperedges x n_nodes)}
        role_order: list[str] roles in deterministic order
        role_weights: torch tensor (n_roles,) weights summing to 1 (or None => uniform)
        dbg_state: dict that may contain:
            - 'printed_role_msg_roles': set()
        """
        device = ego_embeddings.device
        side_embeddings = torch.zeros_like(ego_embeddings)

        # (1) KG adjacency propagation
        if use_kg and (A_in is not None):
            A_in = A_in.to(device)
            side_embeddings = side_embeddings + self._spmm(A_in, ego_embeddings)

        # (2) Role-aware hypergraph propagation
        if use_role_hg and (role_hyper is not None) and (role_order is not None) and (len(role_order) > 0):
            if role_weights is None:
                role_weights = torch.ones(len(role_order), device=device) / float(len(role_order))
            else:
                role_weights = role_weights.to(device)

            printed_roles = None
            if isinstance(dbg_state, dict):
                printed_roles = dbg_state.get("printed_role_msg_roles", None)

            for i, role in enumerate(role_order):
                pack = role_hyper.get(role, None)
                if (pack is None) or (pack.get("norm1", None) is None) or (pack.get("norm2", None) is None):
                    continue

                norm1 = pack["norm1"].to(device)  # (n_nodes x n_hyperedges)
                norm2 = pack["norm2"].to(device)  # (n_hyperedges x n_nodes)

                # hyperedge embeddings: (n_hyperedges x dim) = norm2 @ X
                he = self._spmm(norm2, ego_embeddings)
                # node update: (n_nodes x dim) = norm1 @ he
                node_upd = self._spmm(norm1, he)

                # Runtime safety checks (rare; keep them cheap)
                if torch.isnan(node_upd).any():
                    print(f"[ERROR][Role-HG] NaNs detected in role={role}")

                # One-time debug print per role (prevents log spam)
                if printed_roles is not None and role not in printed_roles:
                    nrm = node_upd.detach().float().norm().item()
                    print(f"[DEBUG][Role-HG MSG] role={role} norm={nrm:.6f}")
                    printed_roles.add(role)

                side_embeddings = side_embeddings + role_weights[i] * node_upd

        # bi-interaction aggregation
        sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
        bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))

        embeddings = (bi_embeddings + sum_embeddings) if self.attention else sum_embeddings
        embeddings = self.message_dropout(embeddings)
        return embeddings


class RAHGREC(nn.Module):
    def __init__(
        self,
        args,
        n_users,
        n_entities,
        n_relations,
        A_in=None,
        user_pre_embed=None,
        item_pre_embed=None,
        h_list=None,
        r_list=None,
        t_list=None,
        role_hyper=None,
        constraint_maps=None,
    ):
        super().__init__()

        self.args = args
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.h_list = h_list
        self.r_list = r_list
        self.t_list = t_list

        # New inputs
        self.role_hyper = role_hyper
        self.constraint_maps = constraint_maps  # stored for evaluation/reranking (if you implement later)

        # Flags
        self.attention = int(getattr(args, "attention", 0))
        self.knowledgegraph = int(getattr(args, "knowledgegraph", 0))
        self.use_role_hg = int(getattr(args, "enable_role_hg", 0)) == 1 and (role_hyper is not None)

        # Basic hyperparams
        self.use_pretrain = args.use_pretrain
        self.embed_dim = args.embed_dim
        self.cf_l2loss_lambda = args.cf_l2loss_lambda
        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.aggregation_type = "bi-interaction"

        # Layers
        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        # Embeddings
        self.entity_user_embed = nn.Embedding(self.n_entities, self.embed_dim)
        nn.init.xavier_uniform_(self.entity_user_embed.weight, gain=nn.init.calculate_gain("relu"))

        self.relation_embed = nn.Embedding(self.n_relations, args.relation_dim)
        nn.init.xavier_uniform_(self.relation_embed.weight, gain=nn.init.calculate_gain("relu"))

        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, args.relation_dim))
        nn.init.xavier_uniform_(self.trans_M, gain=nn.init.calculate_gain("relu"))

        # Aggregator stack
        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(
                HyperAggregator(
                    self.conv_dim_list[k],
                    self.conv_dim_list[k + 1],
                    self.mess_dropout[k],
                    self.aggregation_type,
                    self.attention,
                )
            )

        # A_in (fixed)
        if A_in is None:
            self.A_in = None
        else:
            self.A_in = A_in.coalesce()

        # Role gate
        self.role_order = eval(getattr(args, "roles", "[]")) if self.use_role_hg else []
        self.role_gate = getattr(args, "role_gate", "learned")

        if self.use_role_hg and len(self.role_order) > 0 and self.role_gate == "learned":
            # logits -> softmax weights
            self.role_logits = nn.Parameter(torch.zeros(len(self.role_order)))
        else:
            self.role_logits = None

        # Debug state (prevents log spam)
        self._dbg_rolehg_active_once = False
        self._dbg_state = {"printed_role_msg_roles": set()}
        self._dbg_rolehg_init_once = False

    def _get_role_weights(self):
        if (not self.use_role_hg) or (len(self.role_order) == 0):
            return None
        if self.role_gate == "uniform" or (self.role_logits is None):
            return torch.ones(len(self.role_order), device=self.entity_user_embed.weight.device) / float(len(self.role_order))
        return torch.softmax(self.role_logits, dim=0)

    def _debug_rolehg_init(self):
        """Print once: confirms role_hyper exists and shows H shapes / hyperedge counts."""
        if (not self.use_role_hg) or self._dbg_rolehg_init_once:
            return
        self._dbg_rolehg_init_once = True

        print("[DEBUG][Role-HG INIT] role_order =", self.role_order)
        if self.role_hyper is None:
            print("[DEBUG][Role-HG INIT] role_hyper is None (should not happen if use_role_hg=True)")
            return

        for r in self.role_order:
            pack = self.role_hyper.get(r, None)
            if pack is None or pack.get("H", None) is None:
                print(f"[DEBUG][Role-HG INIT] role={r}: EMPTY (0 triples / mapping mismatch)")
            else:
                H = pack["H"]
                nh = pack.get("n_hyperedges", -1)
                try:
                    shp = tuple(H.shape)
                except Exception:
                    shp = None
                print(f"[DEBUG][Role-HG INIT] role={r}: H shape={shp} hyperedges={nh}")

    def calc_cf_embeddings(self):
        # One-time init dump (useful to confirm role hyperedges were built)
        self._debug_rolehg_init()

        ego_embed = self.entity_user_embed.weight  # (n_entities, dim)
        all_embed = [ego_embed]

        role_w = self._get_role_weights()

        # One-time active print (weights prove gate is used)
        if self.use_role_hg and (not self._dbg_rolehg_active_once):
            print(
                "[DEBUG][Role-HG ACTIVE]",
                "roles =", self.role_order,
                "weights =", role_w.detach().cpu().numpy() if role_w is not None else None
            )
            self._dbg_rolehg_active_once = True

        for layer in self.aggregator_layers:
            ego_embed = layer(
                ego_embeddings=ego_embed,
                A_in=self.A_in,
                role_hyper=self.role_hyper,
                role_order=self.role_order,
                role_weights=role_w,
                use_kg=(self.knowledgegraph == 1),
                use_role_hg=self.use_role_hg,
                dbg_state=self._dbg_state,
            )
            all_embed.append(F.normalize(ego_embed, p=2, dim=1))

        all_embed = torch.cat(all_embed, dim=1)  # (n_entities, concat_dim)
        return all_embed

    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids):
        all_embed = self.calc_cf_embeddings()
        user_embed = all_embed[user_ids]
        item_pos_embed = all_embed[item_pos_ids]
        item_neg_embed = all_embed[item_neg_ids]

        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)

        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        return cf_loss + self.cf_l2loss_lambda * l2_loss

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        r_embed = self.relation_embed(r)
        W_r = self.trans_M[r]

        h_embed = self.entity_user_embed(h)
        pos_t_embed = self.entity_user_embed(pos_t)
        neg_t_embed = self.entity_user_embed(neg_t)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)

        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = (
            _L2_loss_mean(r_mul_h)
            + _L2_loss_mean(r_embed)
            + _L2_loss_mean(r_mul_pos_t)
            + _L2_loss_mean(r_mul_neg_t)
        )
        return kg_loss + self.kg_l2loss_lambda * l2_loss

    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_user_embed.weight[h_list]
        t_embed = self.entity_user_embed.weight[t_list]

        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)

        if self.attention:
            v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        else:
            v_list = torch.sum(r_mul_t, dim=1)
        return v_list

    def update_attention(self, h_list, t_list, r_list, relations):
        if self.A_in is None:
            return

        device = self.A_in.device
        rows, cols, values = [], [], []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in_new = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
        A_in_new = torch.sparse.softmax(A_in_new.cpu(), dim=1).to(device)

        self.A_in = A_in_new.coalesce()

    def calc_score(self, user_ids, item_ids):
        all_embed = self.calc_cf_embeddings()
        user_embed = all_embed[user_ids]
        item_embed = all_embed[item_ids]
        return torch.matmul(user_embed, item_embed.transpose(0, 1))

    def forward(self, *input, mode):
        if mode == "train_cf":
            return self.calc_cf_loss(*input)
        if mode == "train_kg":
            return self.calc_kg_loss(*input)
        if mode == "update_att":
            return self.update_attention(*input)
        if mode == "predict":
            return self.calc_score(*input)
        raise ValueError(f"Unknown mode={mode}")
