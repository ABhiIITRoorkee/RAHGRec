# parser_rahgrec.py
import argparse


def parse_rahgrec_args():
    parser = argparse.ArgumentParser(description="Run RAHGRec.")

    parser.add_argument('--seed', type=int, default=4022, help='Random seed.')

    # dataset path: datasets/<data_name>/<path>/*.txt
    parser.add_argument('--data_dir', nargs='?', default='datasets/', help='Input data path.')
    parser.add_argument('--data_name', nargs='?', default='PyLib', help='Choose a dataset folder name')
    parser.add_argument('--path', nargs='?', default='t06', help='Input valid subfolder name as path.')

    parser.add_argument('--use_pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/', help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth', help='Path of stored model.')

    parser.add_argument('--cf_batch_size', type=int, default=128, help='CF batch size.')
    parser.add_argument('--kg_batch_size', type=int, default=16192, help='KG batch size.')
    parser.add_argument('--test_batch_size', type=int, default=40000, help='Test batch size.')

    parser.add_argument('--embed_dim', type=int, default=128, help='User / entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=64, help='Relation Embedding size.')

    parser.add_argument('--attention', type=int, default=0, help='0: no attention, 1: use attention.')
    parser.add_argument('--knowledgegraph', type=int, default=0, help='0: no KG, 1: use KG.')

    parser.add_argument('--laplacian_type', type=str, default='random-walk',
                        help='Specify adjacency normalization from {symmetric, random-walk}.')

    parser.add_argument('--conv_dim_list', nargs='?', default='[64,64]',
                        help='Output sizes of every aggregation layer.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.01,0.01]',
                        help='Message dropout per layer.')

    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5, help='Lambda for KG l2 loss.')
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5, help='Lambda for CF l2 loss.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')

    parser.add_argument('--n_epoch', type=int, default=1000, help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=50, help='Early stopping window.')
    parser.add_argument('--cf_print_every', type=int, default=10, help='Print CF loss every N iters.')
    parser.add_argument('--kg_print_every', type=int, default=10, help='Print KG loss every N iters.')
    parser.add_argument('--evaluate_every', type=int, default=10, help='Evaluate every N epochs.')

    parser.add_argument('--Ks', nargs='?', default='[5, 10, 15, 20]', help='Evaluate metrics@K.')

    # (legacy) semantic similarity knobs (if you still use them somewhere)
    parser.add_argument('--lambda_p', type=float, default=0.5, help='Semantic weight for project similarity')
    parser.add_argument('--lambda_l', type=float, default=0.5, help='Semantic weight for library similarity')
    parser.add_argument('--alpha_p', type=float, default=0.5, help='Threshold for project similarity')
    parser.add_argument('--alpha_l', type=float, default=0.5, help='Threshold for library similarity')

    # =========================
    # Role-aware Hypergraph (RA-HG)
    # =========================
    parser.add_argument('--enable_role_hg', type=int, default=0,
                        help='1: enable role-aware hypergraph propagation, 0: off.')
    parser.add_argument('--role_source', type=str, default='kg_relation',
                        choices=['kg_relation', 'file_scope'],
                        help='Source of roles for hyperedges.')
    parser.add_argument('--roles', nargs='?', default="[]",
                        help='List of role names, e.g., "[\'dependent_on\',\'described_by\']".')
    parser.add_argument('--role_rel_map', nargs='?', default="{}",
                        help='Dict mapping role -> RAW KG relation id, e.g., "{\'dependent_on\':0}".')
    parser.add_argument('--role_gate', type=str, default='learned',
                        choices=['learned', 'uniform'],
                        help='How to combine role-specific propagations.')
    parser.add_argument('--role_alpha', type=float, default=0.0,
                        help='Optional sparsification; 0 keeps all.')

    # =========================
    # Constraints + adoptability metrics
    # =========================
    parser.add_argument('--enable_constraints', type=int, default=0,
                        help='1: compute constraint/adoptability metrics, 0: off.')
    parser.add_argument('--rel_dep', type=int, default=-1,
                        help='RAW relation id for dependency edges. -1 disables.')
    parser.add_argument('--rel_license', type=int, default=-1,
                        help='RAW relation id for license edges. -1 disables.')
    parser.add_argument('--rel_vuln', type=int, default=-1,
                        help='RAW relation id for vulnerability edges. -1 disables.')
    parser.add_argument('--license_blocklist', nargs='?', default='[]',
                        help='Blocked license node ids, e.g., "[12,15]".')
    parser.add_argument('--block_vulnerable', type=int, default=1,
                        help='1: treat vulnerable libs as violations.')
    parser.add_argument('--enable_rerank', type=int, default=0,
                        help='1: rerank recommendations by constraints, 0: off.')

    args = parser.parse_args()

    save_dir = 'result/{}/edim{}_rdim{}_att{}_kg{}_{}/{}/'.format(
        args.data_name, args.embed_dim, args.relation_dim, args.attention, args.knowledgegraph,
        '-'.join([str(i) for i in eval(args.conv_dim_list)]),
        args.path
    )
    args.save_dir = save_dir
    return args
