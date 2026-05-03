"""
main_enhanced.py — Entry point for enhanced TransFIR.

Usage examples:
  # Baseline (original TransFIR)
  python main_enhanced.py --dataset ICEWS14 --enhancement none

  # V1: Pattern Generation (CVAE)
  python main_enhanced.py --dataset ICEWS14 --enhancement generative

  # V2: Online Adaptation (GRU entity memory)
  python main_enhanced.py --dataset ICEWS14 --enhancement meta
"""

import os
import sys
import pickle
import random
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import utils
from model import sort_by_last_dim_with_neg1_last
from model_enhanced import EnhancedModel


# ---------------------------------------------------------------- logging
class PrintToLog:
    def write(self, message):
        if message != "\n":
            logging.info(message)

    def flush(self):
        pass


def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file, level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console)


sys.stdout = PrintToLog()


# ---------------------------------------------------------------- helpers
def create_timestamped_dir(base_dir, args):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = (
        f"{ts}_dataset_{args.dataset}_enhancement_{args.enhancement}"
        f"_hl_{args.history_len}_ml_{args.max_length}_hd_{args.hidden_dim}"
        f"_nl_{args.num_layers}_nh_{args.num_heads}_nc_{args.num_code}"
        f"_ratio_{args.split_ratio}_tips_{args.tips}"
    )
    d = os.path.join(base_dir, name)
    os.makedirs(d, exist_ok=True)
    return d


def re_split(dataset, split_ratio=(0.5, 0.2, 0.3)):
    all_triples = np.concatenate([dataset.train, dataset.valid, dataset.test])
    times = np.unique(all_triples[:, 3])
    t1 = int(len(times) * split_ratio[0])
    t2 = int(len(times) * (split_ratio[0] + split_ratio[1]))
    dataset.train = all_triples[all_triples[:, 3] <= times[t1 - 1]]
    dataset.valid = all_triples[
        (all_triples[:, 3] > times[t1 - 1]) & (all_triples[:, 3] <= times[t2 - 1])
    ]
    dataset.test = all_triples[all_triples[:, 3] > times[t2 - 1]]
    return dataset


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ---- dataset ----
from torch.utils.data import Dataset


class TKGDataset(Dataset):
    def __init__(self, data_dict, all_triples):
        self.data_dict = data_dict
        self.triples = all_triples
        self.times = np.unique(all_triples[:, 3])

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        t = self.times[idx]
        return torch.tensor(self.triples[self.triples[:, 3] == t])


# ---- embedding init ----
def get_init_embedding(path, n_ent, n_rel, dim, device="cuda", word_embedding=True):
    gamma, eps = 6.0, 1.0
    r = (gamma + eps) / dim
    if word_embedding:
        ent = torch.tensor(np.load(path), dtype=torch.float).to("cuda")
    else:
        ent = nn.Parameter(torch.zeros(n_ent, dim, device=device))
        nn.init.uniform_(ent, -r, r)
    rel = nn.Parameter(torch.zeros(n_rel * 2, dim, device=device))
    nn.init.uniform_(rel, -r, r)
    cls_ = nn.Parameter(torch.zeros(4, dim, device=device))
    nn.init.uniform_(cls_, -r, r)
    miss = nn.Parameter(torch.zeros(1, dim, device=device))
    nn.init.uniform_(miss, -r, r)
    return {
        "entity_embedding": ent,
        "relation_embedding": rel,
        "cls_embedding": cls_,
        "missing_embedding": miss,
    }


# ---- chain helpers ----
def get_topk_chain(query_rel, chain_rel, rel_emb, topk=30):
    q = rel_emb[query_rel].unsqueeze(0)
    c = rel_emb[chain_rel]
    sim = q @ c.T
    _, idx = torch.topk(sim, min(topk, c.shape[0]))
    return idx


def get_embedding_one_hop(head, chain, emb_dict, model, device="cuda"):
    ent_emb = emb_dict["entity_embedding"]
    rel_emb = emb_dict["relation_embedding"]
    empty = emb_dict["missing_embedding"]
    B, N, M = chain.shape
    chain = chain.to(torch.int64)
    chain = sort_by_last_dim_with_neg1_last(chain)
    tp = model.time_projection
    ep = model.entity_down_proj
    rp = model.relation_down_proj
    mask = chain[:, :, 0] == -1
    valid = chain[~mask]
    s, r, o, t = valid[:, 0], valid[:, 1], valid[:, 2], valid[:, 3]
    d = rel_emb.shape[1] // M
    out = torch.zeros(B, N, M, d, device=rel_emb.device)
    out[:] = empty[:, :d]
    ve = torch.zeros(len(valid), M, d, device=rel_emb.device)
    ve[:, 0] = ep(ent_emb[s])
    ve[:, 1] = rp(rel_emb[r])
    ve[:, 2] = ep(ent_emb[o])
    ve[:, 3] = tp(t.unsqueeze(-1).float())
    idx = torch.nonzero(~mask, as_tuple=False)
    out[:, 0, 0] = ep(ent_emb[head])
    out[idx[:, 0], idx[:, 1]] = ve
    out = out.view(B, -1, rel_emb.shape[1])
    return out, mask


def prepare_chains(gt, data, emb_dict, model, max_len=30):
    n = len(data)
    chains = np.zeros((n, max_len, 4)) - 1
    rel_emb = emb_dict["relation_embedding"]
    for i in range(n):
        c = data[i]["one_hop_chain"]
        if len(c) > max_len:
            idx = get_topk_chain(gt[i, 1], c[:, 1], rel_emb, max_len)
            c = c[idx.squeeze().cpu().numpy()]
        chains[i, : len(c), :4] = c
        chains[i, : len(c), 3] = gt[i, 3] - chains[i, : len(c), 3]
    head = gt[:, 0]
    ce, cm = get_embedding_one_hop(
        head, torch.tensor(chains).to("cuda"), emb_dict, model
    )
    return ce, cm, chains



# ================================================================
#  MAIN
# ================================================================
def parse_args():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ICEWS14")
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--history_len", type=int, default=14)
    p.add_argument("--max_length", type=int, default=30)
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--word_embedding", action="store_false", default=True)
    p.add_argument("--word_embedding_path", type=str, default="data")
    p.add_argument("--word_embedding_dim", type=int, default=768)
    p.add_argument("--residual", type=bool, default=True)
    p.add_argument("--result_dir", type=str, default="results_enhanced")
    p.add_argument("--layer_norm", action="store_false", default=True)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--tips", type=str, default="None")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, nargs="+", default=[42])
    p.add_argument("--num_code", type=int, default=50)
    p.add_argument("--ablation", type=str, default="None")
    p.add_argument("--split_ratio", type=int, default=30)
    p.add_argument("--train_horizon_pct", type=int, default=100,
                   help="Keep only the last k%% of the training time window "
                        "(in {25,50,75,100}). 100 = use full horizon.")
    # ---- AdaTKG enhancements ----
    # Public release supports the four AdaTKG variants used in the paper:
    #   meta            -> AdaTKG-GRU      (online GRU adapter)
    #   ema             -> AdaTKG-EMA      (default; learnable EMA)
    #   attention       -> AdaTKG-CrossAtt (cross-attention readout)
    # Plus three ablation-only modes used in Section 4 ablations:
    #   ema_perent      -> AdaTKG-EMA with per-entity decay scalar
    #   ema_perdim      -> AdaTKG-EMA with per-dimension decay vector
    #   ema_constgate   -> AdaTKG-EMA with adaptive gate replaced by g=0.5
    p.add_argument(
        "--enhancement",
        type=str,
        default="ema",
        choices=["none",
                 "meta", "ema", "ema_perent", "ema_perdim", "ema_constgate", "attention"],
        help="Which AdaTKG variant to apply",
    )
    return p.parse_args()


def main(args):
    if os.path.isdir(args.result_dir):
        prior = [
            n for n in os.listdir(args.result_dir)
            if os.path.isdir(os.path.join(args.result_dir, n))
        ]
        if prior:
            print(
                f"[SKIP] {args.result_dir} already has prior run dir(s): {prior}. "
                "Training is in progress or done — exiting."
            )
            return
    results_dir = create_timestamped_dir(args.result_dir, args)
    setup_logging(os.path.join(results_dir, "log.txt"))
    if args.ablation == "no_codebook":
        args.num_code = 1
    args.word_embedding_path = (
        f"{args.word_embedding_path}/{args.dataset}/{args.dataset}_Bert_Entity_Embedding.npy"
    )
    print(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- load data ----
    data = utils.load_data(args.dataset)
    ratio_map = {10: [0.8, 0.1, 0.1], 30: [0.5, 0.2, 0.3],
                 50: [0.3, 0.2, 0.5], 70: [0.2, 0.1, 0.7]}
    split_ratio = ratio_map[args.split_ratio]
    data = re_split(data, split_ratio)

    known_entity = set(data.train[:, 0].tolist() + data.train[:, 2].tolist())
    known_entity_idx = torch.tensor(list(known_entity), dtype=torch.long)
    unknown_entity = set(range(data.num_nodes)) - known_entity
    unk_entity_idx = torch.tensor(list(unknown_entity), dtype=torch.long)
    known_tv = set(
        data.train[:, 0].tolist() + data.train[:, 2].tolist()
        + data.valid[:, 0].tolist() + data.valid[:, 2].tolist()
    )
    known_tv_idx = torch.tensor(list(known_tv), dtype=torch.long)

    all_triple = np.concatenate([data.train, data.valid, data.test])
    times = np.unique(all_triple[:, 3])
    t1 = int(len(times) * split_ratio[0])
    t2 = int(len(times) * (split_ratio[0] + split_ratio[1]))
    train_time = times[:t1]
    valid_time = times[t1:t2]
    test_time = times[t2:-1]

    train_year_start = 0
    train_year_end = len(train_time)
    if getattr(args, "train_horizon_pct", 100) < 100:
        keep = max(1, int(len(train_time) * args.train_horizon_pct / 100.0))
        train_year_start = len(train_time) - keep
        print(f"[train_horizon] keep last {args.train_horizon_pct}% of "
              f"train_time: {keep} time steps "
              f"(start year-index {train_year_start}, "
              f"t={int(train_time[train_year_start])})")

    all_entities = np.unique(np.concatenate([all_triple[:, 0], all_triple[:, 2]]))
    all_relations = np.unique(all_triple[:, 1])
    inv = all_triple[:, [2, 1, 0, 3]].copy()
    inv[:, 1] += len(all_relations)
    all_triple = np.concatenate([all_triple, inv])

    entity_history = {}
    for e in all_entities:
        t = all_triple[all_triple[:, 0] == e]
        entity_history[e] = t[np.argsort(t[:, 3])]

    dataset = TKGDataset(entity_history, all_triple)

    with open(f"data/{args.dataset}/{args.dataset}_T_{args.history_len}.pkl", "rb") as f:
        history_dataset = pickle.load(f)

    # ---- run per seed ----
    best_result_dict = {}
    for seed in args.seed:
        set_random_seed(seed)
        print(f"\n===== Seed {seed}  Enhancement: {args.enhancement} =====")

        use_we = args.ablation not in ["no_word_embedding", "no_ITC"]
        emb_dict = get_init_embedding(
            args.word_embedding_path, data.num_nodes, data.num_rels,
            args.hidden_dim, device=device, word_embedding=use_we,
        )

        model = EnhancedModel(
            data.num_nodes, data.num_rels,
            num_heads=args.num_heads,
            entity_dim=args.hidden_dim,
            relation_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            word_embedding=args.word_embedding,
            word_embedding_path=args.word_embedding_path,
            layer_norm=args.layer_norm,
            word_embedding_dim=args.word_embedding_dim,
            num_code=args.num_code,
            ablation=args.ablation if args.ablation != "None" else None,
            enhancement=args.enhancement,
        ).to(device)

        if args.ablation == "no_word_embedding":
            all_params = list(model.parameters()) + [
                emb_dict["relation_embedding"], emb_dict["cls_embedding"],
                emb_dict["missing_embedding"], emb_dict["entity_embedding"],
            ]
        else:
            all_params = list(model.parameters()) + [
                emb_dict["relation_embedding"], emb_dict["cls_embedding"],
                emb_dict["missing_embedding"],
            ]
        optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=1e-5)

        best_val_mrr = 0
        patience = 0
        best_test_metric = ""
        best_result = {}

        for epoch in range(args.epochs):
            model.train()

            # V2: reset entity memory each epoch
            model.reset_online_memory()

            train_loss_list, rank_list = [], []

            for year in tqdm(range(train_year_start, train_year_end), desc=f"Epoch {epoch} train"):
                data_year = dataset[year]
                hist = history_dataset[year]
                for batch_id in range(2):
                    s = int(batch_id * len(data_year) / 2)
                    e = int((batch_id + 1) * len(data_year) / 2)
                    ce, cm, raw_chains = prepare_chains(data_year, hist, emb_dict, model,
                                               max_len=args.max_length)
                    tri = data_year[s:e].to(device)
                    ce_b = ce[s:e]
                    cm_b = cm[s:e]

                    score, loss = model(tri, ce_b, cm_b, emb_dict,
                                        chain_meta=None,
                                        epoch=epoch, max_epochs=args.epochs)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    train_loss_list.append(loss.item())
                    rank_list.append(utils.get_rank(score, tri[:, 2]))

            mrr, h1, h3, h10 = utils.get_metric(rank_list)
            avg_loss = np.mean(train_loss_list)
            print(f"[Train] epoch={epoch} loss={avg_loss:.4f} MRR={mrr:.4f} H@1={h1:.4f} H@3={h3:.4f} H@10={h10:.4f}")

            # ---- Validation ----
            model.eval()
            rank_list, valid_triples_all = [], []
            with torch.no_grad():
                for year in range(len(train_time), len(train_time) + len(valid_time)):
                    data_year = dataset[year]
                    hist = history_dataset[year]
                    batch_tri = np.zeros((0, 4))
                    for bid in range(2):
                        s = int(bid * len(data_year) / 2)
                        e = int((bid + 1) * len(data_year) / 2)
                        ce, cm, _ = prepare_chains(data_year, hist, emb_dict, model,
                                                   max_len=args.max_length)
                        tri = data_year[s:e].to(device)
                        score, _ = model(tri, ce[s:e], cm[s:e], emb_dict)
                        rank_list.append(utils.get_rank(score, tri[:, 2]).cpu())
                        batch_tri = np.concatenate((batch_tri, tri.cpu().numpy()))
                    valid_triples_all.append(batch_tri)

            mrr, h1, h3, h10 = utils.get_metric(rank_list)
            if epoch == 0:
                v_unk_idx = utils.get_unkown_index(valid_triples_all, unk_entity_idx)
                v_emg_idx = utils.get_emerging_index(valid_triples_all, known_entity_idx)
            e_mrr, e_h1, e_h3, e_h10 = utils.get_metric_emerging_both(
                rank_list, valid_triples_all, v_emg_idx
            )
            print(f"[Valid] MRR={mrr:.4f} H@1={h1:.4f} H@3={h3:.4f} H@10={h10:.4f}")
            print(f"[Valid Emerging] MRR={e_mrr:.4f} H@1={e_h1:.4f} H@3={e_h3:.4f} H@10={e_h10:.4f}")

            # ---- Test (only when val improves) ----
            if e_mrr > best_val_mrr:
                best_val_mrr = e_mrr
                patience = 0
                rank_list, test_triples_all = [], []
                with torch.no_grad():
                    for year in range(len(train_time) + len(valid_time),
                                     len(train_time) + len(valid_time) + len(test_time) - 1):
                        data_year = dataset[year]
                        hist = history_dataset[year]
                        batch_tri = np.zeros((0, 4))
                        for bid in range(2):
                            s = int(bid * len(data_year) / 2)
                            e = int((bid + 1) * len(data_year) / 2)
                            ce, cm, _ = prepare_chains(data_year, hist, emb_dict, model,
                                                       max_len=args.max_length)
                            tri = data_year[s:e].to(device)
                            score, _ = model(tri, ce[s:e], cm[s:e], emb_dict)
                            rank_list.append(utils.get_rank(score, tri[:, 2]).cpu())
                            batch_tri = np.concatenate((batch_tri, tri.cpu().numpy()))
                        test_triples_all.append(batch_tri)
                    if epoch == 0:
                        t_unk_idx = utils.get_unkown_index(test_triples_all, unk_entity_idx)
                        t_emg_idx = utils.get_emerging_index(test_triples_all, known_tv_idx)
                    tmrr, th1, th3, th10 = utils.get_metric(rank_list)
                    te_mrr, te_h1, te_h3, te_h10 = utils.get_metric_emerging_both(
                        rank_list, test_triples_all, t_emg_idx
                    )
                    tu_mrr, tu_h1, tu_h3, tu_h10 = utils.get_metric_unknown_both(
                        rank_list, test_triples_all, t_unk_idx
                    )
                    print(f"[Test] MRR={tmrr:.4f} H@1={th1:.4f} H@3={th3:.4f} H@10={th10:.4f}")
                    print(f"[Test Emerging] MRR={te_mrr:.4f} H@1={te_h1:.4f} H@3={te_h3:.4f} H@10={te_h10:.4f}")
                    print(f"[Test Unknown] MRR={tu_mrr:.4f} H@1={tu_h1:.4f} H@3={tu_h3:.4f} H@10={tu_h10:.4f}")
                    best_test_metric = (
                        f"Emg MRR={te_mrr:.4f} H@1={te_h1:.4f} H@3={te_h3:.4f} H@10={te_h10:.4f}"
                    )
                    best_result = {
                        "all_mrr": tmrr, "all_hit1": th1, "all_hit3": th3, "all_hit10": th10,
                        "unknown_mrr": tu_mrr, "unknown_hit1": tu_h1,
                        "unknown_hit3": tu_h3, "unknown_hit10": tu_h10,
                        "emerging_mrr": te_mrr, "emerging_hit1": te_h1,
                        "emerging_hit3": te_h3, "emerging_hit10": te_h10,
                    }
                    torch.save(model.state_dict(), os.path.join(results_dir, f"model_seed{seed}.pth"))
                    torch.save(emb_dict, os.path.join(results_dir, f"emb_seed{seed}.pth"))
            else:
                patience += 1

            if patience > args.patience:
                print(f"Early stop at epoch {epoch}, best val MRR={best_val_mrr:.4f}")
                print(f"Best test: {best_test_metric}")
                best_result_dict[seed] = best_result
                break
            if epoch == args.epochs - 1:
                best_result_dict[seed] = best_result

    # ---- summary ----
    print("\n===== Final Results =====")
    for seed, r in best_result_dict.items():
        print(
            f"Seed {seed}: All MRR={r['all_mrr']:.4f} "
            f"Emg MRR={r['emerging_mrr']:.4f} H@3={r['emerging_hit3']:.4f} H@10={r['emerging_hit10']:.4f}"
        )
    if best_result_dict:
        avg = {k: np.mean([r[k] for r in best_result_dict.values()]) for k in best_result.keys()}
        print(
            f"Avg: All MRR={avg['all_mrr']:.4f} "
            f"Emg MRR={avg['emerging_mrr']:.4f} H@3={avg['emerging_hit3']:.4f} H@10={avg['emerging_hit10']:.4f}"
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
