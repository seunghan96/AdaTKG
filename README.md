# AdaTKG

Reference implementation of **AdaTKG**, the temporal knowledge graph
(TKG) reasoning method introduced in our paper. AdaTKG augments the
static-inductive baseline TransFIR with a per-entity online memory
governed by an **adaptive gate**, refining each entity's representation
every time the entity participates in a fact.

The repository supports four models reported in the paper:

| Model | `--enhancement` | Description |
|---|---|---|
| **Base**            | `none`       | TransFIR backbone (no per-entity memory). |
| **AdaTKG-EMA**      | `ema`        | Default: learnable EMA with a single shared scalar. |
| **AdaTKG-GRU**      | `meta`       | Online GRU adapter. |
| **AdaTKG-CrossAtt** | `attention`  | Cross-attention readout over a bounded per-entity buffer. |

All four share the same TransFIR backbone (BERT static encoder, VQ
codebook, ConvTransE decoder) and only differ in the memory update rule.

---

## 1. Installation

```bash
conda create -n adatkg python=3.10 -y
conda activate adatkg
pip install -r requirements.txt
```

A single NVIDIA A100 80GB is sufficient for every experiment in the
paper. CPU inference is not supported.

---

## 2. Datasets

We use the four standard TKG benchmarks **ICEWS14**, **ICEWS18**,
**ICEWS05-15**, and **GDELT** under the same setup as
TransFIR~\[1\]. Datasets are **not** shipped with the repo (`data/`
is `.gitignore`d); the user prepares them once locally before training.

### 2.1 Expected file layout

```
data/
├── ICEWS14/
│   ├── train.txt          ← tab-separated quadruples (subj  rel  obj  time_idx)
│   ├── valid.txt
│   ├── test.txt
│   ├── entity2id.txt      ← line i (0-indexed) = entity surface form, then "\t" + id
│   └── relation2id.txt
├── ICEWS18/                ← same five files
├── ICEWS05-15/             ← same five files
└── GDELT/                  ← same five files (entity surface form ends with " (...)" sense suffix)
```

Each line of `train.txt` / `valid.txt` / `test.txt` is `subj_id<TAB>rel_id<TAB>obj_id<TAB>time_idx`,
with `time_idx` an integer index in chronological order (1 step = 1 day
for ICEWS, 15 minutes for GDELT). `entity2id.txt` and
`relation2id.txt` map the integer ids back to surface forms.

### 2.2 Where to get the raw files

The four benchmarks are publicly redistributed by the authors of
TransFIR and earlier TKG works~\[1, 2\]. We adopt the **TransFIR
release** verbatim (same files, same chronological 5:2:3 split):

- TransFIR official repository (datasets folder):
  `https://github.com/<TransFIR-authors>/TransFIR` (please refer to
  the dataset-download instructions in their `README` and copy the
  resulting `train.txt / valid.txt / test.txt / entity2id.txt /
  relation2id.txt` directly into our `data/<DATASET>/` folders).
- Original sources: ICEWS14 / ICEWS18 / ICEWS05-15 are derived from
  the ICEWS event corpus on Harvard Dataverse~\[1\]; GDELT is from
  the GDELT event database~\[2\].

\[1\] Boschee et al., *ICEWS Coded Event Data*, Harvard Dataverse, 2015.
\[2\] Leetaru and Schrodt, *GDELT: Global Data on Events, Location and
Tone*, ISA Annual Convention, 2013.

### 2.3 One-time preprocessing

Once `data/<DATASET>/{train,valid,test,entity2id,relation2id}.txt` are
in place, the training launcher auto-builds two caches per dataset:

- `data/<DS>/<DS>_T_14.pkl` — interaction-chain cache built by
  `data_process.py`.
- `data/<DS>/<DS>_Bert_Entity_Embedding.npy` — frozen BERT embeddings
  built by `word_embedding.py`.

You can also build them manually:

```bash
# Interaction-chain cache (T = history window in days)
python3 data_process.py    --dataset ICEWS14 --T 14

# BERT [CLS] embeddings (set --bert_model_path to your local
# bert-base-uncased checkpoint, or any HuggingFace BERT compatible model)
python3 word_embedding.py  --dataset ICEWS14 \
                           --bert_model_path bert-base-uncased
```

Repeat per dataset. The two caches are reused across all four models
(`Base`, `AdaTKG-EMA`, `AdaTKG-GRU`, `AdaTKG-CrossAtt`) and across HP
configurations, so this preprocessing step happens only once per
benchmark.

---

## 3. Quick start (best configuration)

The launcher `run_experiment.sh` reads the per-(model, dataset)
**best hyperparameter configuration** from `best_configs/<MODEL>.csv`
(the same values reported in Appendix C of the paper) and trains with
that single configuration:

```bash
# AdaTKG-EMA on ICEWS14 (default best HP), GPU 0
bash run_experiment.sh AdaTKG-EMA train ICEWS14 0

# AdaTKG-GRU on ICEWS18, GPU 1
bash run_experiment.sh AdaTKG-GRU train ICEWS18 1

# AdaTKG-CrossAtt on GDELT, GPU 0
bash run_experiment.sh AdaTKG-CrossAtt train GDELT 0

# Base (TransFIR) on ICEWS05-15, GPU 0
bash run_experiment.sh Base train ICEWS05-15 0
```

Logs land in `${SAVE_ROOT:-./results}/<EXP_ID>/train_log.txt`. The
final emerging-slice MRR / Hits@k can be read off the last
`[Test Emerging]` line of the log.

To override the best-HP configuration, pass HP env vars before the
command:

```bash
ML=15 NL=2 HD=512 NC=50 bash run_experiment.sh AdaTKG-EMA train ICEWS14 0
```

---

## 4. Best hyperparameter configurations

`best_configs/<MODEL>.csv` lists the per-dataset selected HP values:

| File | Model |
|---|---|
| `best_configs/Base.csv`            | TransFIR baseline (= AdaTKG-EMA's best HP). |
| `best_configs/AdaTKG-EMA.csv`      | AdaTKG-EMA (default operator). |
| `best_configs/AdaTKG-GRU.csv`      | AdaTKG-GRU. |
| `best_configs/AdaTKG-CrossAtt.csv` | AdaTKG-CrossAtt. |

Each row has columns `model, dataset, max_length, num_layers, hidden_dim, num_code`.
The Base baseline is trained at the same HP as AdaTKG-EMA so that the
efficiency comparison in Table 7 is apples-to-apples.

---

## 5. Repository layout

```
AdaTKG/
├── README.md                 ← this file
├── requirements.txt
├── run_experiment.sh         ← single launcher for all four models
├── best_configs/             ← per-(model, dataset) best HP CSVs
│   ├── Base.csv
│   ├── AdaTKG-EMA.csv
│   ├── AdaTKG-GRU.csv
│   └── AdaTKG-CrossAtt.csv
├── data/                     ← raw datasets + preprocessing caches
├── data_process.py           ← builds interaction-chain cache
├── word_embedding.py         ← builds frozen BERT entity embeddings
├── main.py                   ← Base (TransFIR) training entry point
├── main_enhanced.py          ← AdaTKG variants entry point
├── model.py                  ← TransFIR backbone (frozen, unchanged)
├── model_enhanced.py         ← AdaTKG model wrapper
├── modules_enhanced.py       ← OnlineAdapter / EMAAdapter / AttentionAdapter
├── knowledge_graph.py
└── utils.py
```

---

## 6. Reproducing the main-paper results

Reproducing Tables 1–3 (main results) reduces to one launcher call per
`(model, dataset)`:

```bash
for DS in ICEWS14 ICEWS18 ICEWS05-15 GDELT; do
    bash run_experiment.sh Base            train ${DS} 0
    bash run_experiment.sh AdaTKG-EMA      train ${DS} 0
    bash run_experiment.sh AdaTKG-GRU      train ${DS} 0
    bash run_experiment.sh AdaTKG-CrossAtt train ${DS} 0
done
```

After training, run the same launcher with `test` to extract the
emerging-slice metrics:

```bash
bash run_experiment.sh AdaTKG-EMA test ICEWS14 0
```


---
