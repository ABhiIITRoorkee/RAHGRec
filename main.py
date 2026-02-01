# ===========================================================
#  This source file is based on the PyRec framework published by Bo Li et al.
#  We would like to thank and offer our appreciation to them.
# ===========================================================



# main.py

import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import torch

import RAHGREC  # IMPORTANT: this is a module, so class is RAHGREC.RAHGREC

from parser_rahgrec import *
from log_helper import *
from metrics import *
from model_helper import *
from loader_rahgrec import DataLoaderRAHGREC

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def evaluate(model, dataloader, Ks, device, filename):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['precision', 'recall', 'fone', 'ndcg', 'map', 'mrr']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)
            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, mode='predict')

            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(
                batch_scores,
                train_user_dict,
                test_user_dict,
                batch_user_ids.cpu().numpy(),
                item_ids.cpu().numpy(),
                Ks,
                filename
            )

            cf_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])

            pbar.update(1)

    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.mean(metrics_dict[k][m])

    return cf_scores, metrics_dict


def train(args):
    filename = os.path.join(args.data_dir, args.path[:3] + "_recommendation")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    print('device:', device, 'n_gpu:', n_gpu)

    print('load data ...')
    data = DataLoaderRAHGREC(args, logging)

    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # -------------------- MODEL --------------------
    model = RAHGREC.RAHGREC(
        args,
        data.n_users,
        data.n_entities,
        data.n_relations,
        data.A_in,
        user_pre_embed,
        item_pre_embed,
        data.h_list,
        data.r_list,
        data.t_list,
        role_hyper=getattr(data, "role_hyper", None),
        constraint_maps=getattr(data, "constraint_maps", None),
    )
    model.to(device)

    cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_epoch = -1
    best_recall = 0

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'fone': [], 'ndcg': [], 'map': [], 'mrr': []} for k in Ks}

    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()

        # -------------------- CF TRAIN --------------------
        time1 = time()
        cf_total_loss = 0.0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for it in range(1, n_cf_batch + 1):
            time2 = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(
                data.train_user_dict, data.cf_batch_size
            )
            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)

            cf_optimizer.zero_grad(set_to_none=True)
            cf_batch_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, mode='train_cf')

            if torch.isnan(cf_batch_loss):
                logging.info(
                    'ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, it, n_cf_batch)
                )
                sys.exit()

            cf_batch_loss.backward()
            cf_optimizer.step()

            cf_total_loss += cf_batch_loss.item()

            if (it % args.cf_print_every) == 0:
                logging.info(
                    'CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                        epoch, it, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / it
                    )
                )

        logging.info(
            'CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(
                epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch
            )
        )

        # -------------------- KG TRAIN --------------------
        time3 = time()
        kg_total_loss = 0.0
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1

        for it in range(1, n_kg_batch + 1):
            time4 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(
                data.train_kg_dict, data.kg_batch_size, data.n_users_entities
            )
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            kg_optimizer.zero_grad(set_to_none=True)
            kg_batch_loss = model(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, mode='train_kg')

            if torch.isnan(kg_batch_loss):
                logging.info(
                    'ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, it, n_kg_batch)
                )
                sys.exit()

            kg_batch_loss.backward()
            kg_optimizer.step()

            kg_total_loss += kg_batch_loss.item()

            if (it % args.kg_print_every) == 0:
                logging.info(
                    'KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                        epoch, it, n_kg_batch, time() - time4, kg_batch_loss.item(), kg_total_loss / it
                    )
                )

        logging.info(
            'KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(
                epoch, n_kg_batch, time() - time3, kg_total_loss / n_kg_batch
            )
        )

        # -------------------- UPDATE ATTENTION (NO GRAD) --------------------
        if args.attention == 1 and getattr(model, "A_in", None) is not None:
            time5 = time()
            model.eval()
            with torch.no_grad():
                h_list = data.h_list.to(device)
                t_list = data.t_list.to(device)
                r_list = data.r_list.to(device)
                relations = list(data.laplacian_dict.keys())
                model(h_list, t_list, r_list, relations, mode='update_att')

            logging.info('Update Attention: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time5))

        logging.info('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

        # -------------------- EVAL --------------------
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time6 = time()
            _, metrics_dict = evaluate(model, data, Ks, device, filename)
            logging.info(
                'CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], '
                'F1 [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}], Map [{:.4f}, {:.4f}], Mrr [{:.4f}, {:.4f}]'.format(
                    epoch, time() - time6,
                    metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'],
                    metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'],
                    metrics_dict[k_min]['fone'], metrics_dict[k_max]['fone'],
                    metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg'],
                    metrics_dict[k_min]['map'], metrics_dict[k_max]['map'],
                    metrics_dict[k_min]['mrr'], metrics_dict[k_max]['mrr']
                )
            )

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'fone', 'ndcg', 'map', 'mrr']:
                    metrics_list[k][m].append(metrics_dict[k][m])

            best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)
            if should_stop:
                break

            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                best_epoch = epoch

    # -------------------- SAVE METRICS --------------------
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for k in Ks:
        for m in ['precision', 'recall', 'fone', 'ndcg', 'map', 'mrr']:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))

    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + 'metrics.csv', sep='\t', index=False)

    best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
    logging.info(
        'Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], '
        'F1 [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}], Map [{:.4f}, {:.4f}], Mrr [{:.4f}, {:.4f}]'.format(
            int(best_metrics['epoch_idx']),
            best_metrics['precision@{}'.format(k_min)], best_metrics['precision@{}'.format(k_max)],
            best_metrics['recall@{}'.format(k_min)], best_metrics['recall@{}'.format(k_max)],
            best_metrics['fone@{}'.format(k_min)], best_metrics['fone@{}'.format(k_max)],
            best_metrics['ndcg@{}'.format(k_min)], best_metrics['ndcg@{}'.format(k_max)],
            best_metrics['map@{}'.format(k_min)], best_metrics['map@{}'.format(k_max)],
            best_metrics['mrr@{}'.format(k_min)], best_metrics['mrr@{}'.format(k_max)],
        )
    )

    best_metrics_output = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch]
    print(best_metrics_output)
    best_metrics_output.to_csv(args.save_dir + 'result.csv', sep='\t', index=False)


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = DataLoaderRAHGREC(args, logging)

    model = RAHGREC.RAHGREC(
        args,
        data.n_users,
        data.n_entities,
        data.n_relations,
        data.A_in,
        None,
        None,
        data.h_list,
        data.r_list,
        data.t_list,
        role_hyper=getattr(data, "role_hyper", None),
        constraint_maps=getattr(data, "constraint_maps", None),
    )

    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    filename = os.path.join(args.data_dir, args.path[:3] + "_recommender")
    cf_scores, metrics_dict = evaluate(model, data, Ks, device, filename)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores)

    print(
        'CF Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
            metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'],
            metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'],
            metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']
        )
    )


if __name__ == '__main__':
    args = parse_rahgrec_args()
    train(args)
    # predict(args)
