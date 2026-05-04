"""
AdaTKG — TransFIR backbone augmented with a per-entity online memory.

Available --enhancement modes:
  --- main (paper's three operators) ---
    none            Base TransFIR (no per-entity memory)
    meta            AdaTKG-GRU      (online GRU adapter)
    ema             AdaTKG-EMA      (default; learnable EMA)
    attention       AdaTKG-CrossAtt (cross-attention readout)
  --- ablation-only EMA variants (Section 4 of the paper) ---
    ema_perent      AdaTKG-EMA with per-entity decay scalar
    ema_perdim      AdaTKG-EMA with per-dimension decay vector
    ema_constgate   AdaTKG-EMA with adaptive gate replaced by g=0.5
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import (
    PositionalEncoding,
    TransformerEncoder_with_query,
    VectorQuantizerEMA,
    ConvTransE,
    BasicDriftMLP,
    CodebookDecoder,
)
from modules_enhanced import (
    OnlineAdapter,
    EMAAdapter, AttentionAdapter,
)


class EnhancedModel(nn.Module):
    """TransFIR backbone with the four AdaTKG variants used in the paper:
        - meta            -> AdaTKG-GRU      (online GRU adapter)
        - ema             -> AdaTKG-EMA      (default; learnable EMA)
        - attention       -> AdaTKG-CrossAtt (cross-attention readout)
    Plus three EMA ablation modes (Section 4 of the paper):
        - ema_perent      -> AdaTKG-EMA with per-entity decay scalar
        - ema_perdim      -> AdaTKG-EMA with per-dimension decay vector
        - ema_constgate   -> AdaTKG-EMA with adaptive gate replaced by g=0.5
    """

    VALID_ENHANCEMENTS = {
        "none",
        "meta", "ema", "ema_perent", "ema_perdim", "ema_constgate", "attention",
    }

    def __init__(
        self,
        num_ent,
        num_rel,
        num_heads,
        entity_dim,
        relation_dim,
        num_layers,
        dropout=0.0,
        word_embedding_path=None,
        word_embedding=False,
        residual=True,
        device="cuda",
        layer_norm=False,
        chain_max_length=10,
        time_length=14,
        word_embedding_dim=768,
        num_code=50,
        ablation=None,
        enhancement="none",
        update_timing="before",
    ):
        super().__init__()
        assert enhancement in self.VALID_ENHANCEMENTS, (
            f"enhancement must be one of {self.VALID_ENHANCEMENTS}"
        )
        self.enhancement = enhancement
        self.update_timing = update_timing

        self.activation = F.relu
        self.word_embedding = word_embedding
        self.residual = residual
        self.num_rel = num_rel
        self.num_ents = num_ent
        num_rel_doubled = num_rel * 2
        self.layer_norm = layer_norm
        self.device = device
        self.entity_dim = entity_dim
        self.chain_max_length = chain_max_length
        self.num_code = num_code
        self.time_length = time_length
        self.ablation = ablation

        # ---- shared layers (identical to MyModel) ----
        self.weight_t = nn.Parameter(torch.randn(1, entity_dim))
        self.bias_t = nn.Parameter(torch.randn(1, entity_dim))
        self.bn_entity = nn.BatchNorm1d(entity_dim)
        self.bn_relation = nn.BatchNorm1d(relation_dim)
        self.bn_1 = nn.BatchNorm1d(entity_dim)
        self.bn_2 = nn.BatchNorm1d(entity_dim)

        self.entity_down_proj = nn.Linear(word_embedding_dim, entity_dim // 4)
        self.relation_down_proj = nn.Linear(relation_dim, relation_dim // 4)
        self.time_projection = nn.Linear(1, entity_dim // 4)

        if word_embedding:
            self.entity_embedding = torch.tensor(
                np.load(word_embedding_path), dtype=torch.float
            ).to("cuda")
        else:
            ent_param = nn.Parameter(torch.Tensor(num_ent, entity_dim))
            nn.init.xavier_uniform_(ent_param, gain=nn.init.calculate_gain("relu"))
            self.entity_embedding = ent_param.to(device)
        if word_embedding and entity_dim != word_embedding_dim:
            self.project = nn.Linear(word_embedding_dim, entity_dim)
        else:
            self.project = nn.Linear(entity_dim, entity_dim)

        self.relation_embedding = nn.Parameter(
            torch.Tensor(num_rel_doubled, relation_dim)
        ).to(device)
        self.empty_embedding = nn.Parameter(torch.Tensor(1, entity_dim)).to(device)
        self.cls_embedding = nn.Parameter(torch.Tensor(4, entity_dim)).to(device)
        self.filling_embedding = nn.Parameter(torch.Tensor(1, entity_dim)).to(device)

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.empty_embedding, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.cls_embedding, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.filling_embedding, gain=nn.init.calculate_gain("relu"))

        self.merge_layer = nn.Linear(entity_dim + relation_dim, entity_dim)
        self.encoder = TransformerEncoder_with_query(
            num_layers, entity_dim, num_heads, entity_dim * 2, dropout
        )
        self.w = nn.Linear(entity_dim * 2, entity_dim)
        self.w2 = nn.Linear(entity_dim, 1)
        self.w4 = nn.Linear(entity_dim * 2, 1)
        self.scoring_layer1 = nn.Linear(entity_dim, entity_dim)
        self.scoring_layer2 = nn.Linear(entity_dim, 1)
        self.projection = nn.Linear(entity_dim * 2, entity_dim)
        self.relation_proj = nn.Linear(relation_dim, relation_dim)
        self.lstm_encoder = nn.LSTM(entity_dim, entity_dim, batch_first=True)
        self.mlp_encoder_1 = nn.Linear(entity_dim * 4, entity_dim * 2)
        self.mlp_encoder_2 = nn.Linear(entity_dim * 2, entity_dim)

        self.decoder = ConvTransE(
            num_ent, entity_dim,
            input_dropout=dropout, hidden_dropout=dropout, feature_map_dropout=dropout,
        )
        self.VQDecoder = CodebookDecoder(num_codes=num_code, embedding_dim=entity_dim)
        self.VQ = VectorQuantizerEMA(
            num_codes=num_code, embedding_dim=entity_dim,
            commitment_beta=0.25, decay=0.99, usage_lambda=5e-3,
        )

        # ---- Drift module ----
        self.Drift = BasicDriftMLP(entity_dim)

        # ---- AdaTKG online memory adapter ----
        if enhancement == "meta":
            self.online_adapter = OnlineAdapter(entity_dim, num_ent)
        elif enhancement == "ema":
            self.online_adapter = EMAAdapter(entity_dim, num_ent, decay_mode="shared",
                                             update_timing=update_timing)
        elif enhancement == "ema_perent":
            self.online_adapter = EMAAdapter(entity_dim, num_ent, decay_mode="perentity",
                                             update_timing=update_timing)
        elif enhancement == "ema_perdim":
            self.online_adapter = EMAAdapter(entity_dim, num_ent, decay_mode="perdim",
                                             update_timing=update_timing)
        elif enhancement == "ema_constgate":
            self.online_adapter = EMAAdapter(entity_dim, num_ent,
                                             decay_mode="shared",
                                             gate_mode="constant", const_gate=0.5,
                                             update_timing=update_timing)
        elif enhancement == "attention":
            self.online_adapter = AttentionAdapter(entity_dim, num_ent)

        print(f"[EnhancedModel] enhancement={enhancement}")
        print(f"  entity_embedding shape: {self.entity_embedding.shape}")
        print(f"  relation_embedding shape: {self.relation_embedding.shape}")

    def reset_online_memory(self):
        if self.enhancement in ("meta", "ema", "ema_perent", "ema_perdim",
                                 "ema_constgate", "attention"):
            self.online_adapter.reset_memory()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, triples, chain_embedding, chain_mask, embedding_dict,
                chain_meta=None, epoch=0, max_epochs=100):
        """
        Args
            triples:         (B, 4) — head, relation, tail, time
            chain_embedding: (B, L, D)
            chain_mask:      (B, L)
            embedding_dict:  dict
            chain_meta:      unused (kept for backward-compatible signature)
            epoch/max_epochs: unused
        """
        torch.autograd.set_detect_anomaly(True)
        ground_truth = triples
        query_relation_embedding = embedding_dict["relation_embedding"][ground_truth[:, 1]]
        extra_loss = triples.new_tensor(0.0, dtype=torch.float)

        # ---- 1. Interaction Chain Encoding ----
        if self.ablation == "no_ITC":
            query_embedding = embedding_dict["entity_embedding"][ground_truth[:, 0]]
        else:
            chain_embedding = self.encoder(chain_embedding)
            chain_weight = chain_embedding + query_relation_embedding.unsqueeze(1)
            chain_weight = chain_weight.reshape(-1, self.entity_dim)
            att = F.softmax(self.w2(chain_weight).reshape(chain_embedding.shape[0], -1), dim=1)
            chain_embedding_flat = att.unsqueeze(-1) * chain_embedding
            chain_embedding = torch.mean(chain_embedding_flat, dim=1)
            if self.ablation == "no_batchnorm":
                query_embedding = chain_embedding
            else:
                query_embedding = self.bn_1(chain_embedding)

        query_entity = ground_truth[:, 0]
        edge_type = ground_truth[:, 1]
        label = ground_truth[:, 2]

        relation = self.bn_relation(query_relation_embedding)
        query_embedding = query_embedding + relation

        static_entity_embedding = self.project(self.entity_embedding)

        if self.ablation == "ITC":
            static_entity_embedding = embedding_dict["entity_embedding"][ground_truth[:, 0]]
        else:
            static_entity_embedding[query_entity] = (
                static_entity_embedding[query_entity] + query_embedding
            )

        # ---- 2. VQ Codebook ----
        quant, pseudo_onto_idx, vq_loss, aux = self.VQ(static_entity_embedding)
        query_onto = pseudo_onto_idx[query_entity]

        # ---- 3. Cluster-level drift ----
        drift_embedding = torch.zeros(self.num_code, self.entity_dim, device=self.device)
        drift_embedding.index_add_(0, query_onto, query_embedding)
        count = torch.bincount(query_onto, minlength=self.num_code).unsqueeze(-1).clamp(min=1)
        drift_embedding = drift_embedding / count
        drift_embedding = drift_embedding[pseudo_onto_idx]

        entity_onto_embedding = static_entity_embedding[pseudo_onto_idx]

        # ---- 4. Drift computation + per-entity memory adaptation ----
        if self.enhancement in ("meta", "ema", "ema_perent", "ema_perdim", "ema_constgate"):
            drift_input = torch.cat([static_entity_embedding, entity_onto_embedding], dim=1)
            drift_weight = self.Drift(drift_input)
            adapted_query = self.online_adapter(
                query_entity,
                static_entity_embedding[query_entity].detach(),
                chain_embs=query_embedding.detach(),
                relation_embs=relation.detach(),
            )
            static_entity_embedding = static_entity_embedding.clone()
            static_entity_embedding[query_entity] = adapted_query

        elif self.enhancement == "attention":
            drift_input = torch.cat([static_entity_embedding, entity_onto_embedding], dim=1)
            drift_weight = self.Drift(drift_input)
            adapted_query = self.online_adapter(
                query_entity,
                static_entity_embedding[query_entity].detach(),
                chain_embs=query_embedding.detach(),
                relation_embs=relation.detach(),
                query_embs=query_embedding.detach(),
            )
            static_entity_embedding = static_entity_embedding.clone()
            static_entity_embedding[query_entity] = adapted_query

        else:
            # enhancement == "none": Base TransFIR forward, no per-entity memory.
            drift_input = torch.cat([static_entity_embedding, entity_onto_embedding], dim=1)
            drift_weight = self.Drift(drift_input)

        # ---- 5. Dynamic entity embedding ----
        query_entity_unique = query_entity.unique()
        not_query_mask = torch.ones(self.num_ents, dtype=torch.bool, device=self.device)
        not_query_mask[query_entity_unique] = False

        dynamic_entity_embedding = static_entity_embedding.clone()

        if self.ablation in ["no_drift", "no_codebook"]:
            dynamic_entity_embedding = self.project(self.entity_embedding)
        else:
            dynamic_entity_embedding[not_query_mask] = (
                dynamic_entity_embedding[not_query_mask]
                + (drift_weight * drift_embedding)[not_query_mask]
            )

        if self.layer_norm:
            dynamic_entity_embedding = F.normalize(dynamic_entity_embedding)

        # ---- 6. Scoring ----
        scores_ob, _ = self.decoder(
            query_embedding, self.relation_embedding,
            dynamic_entity_embedding, query_entity, edge_type,
        )

        if self.ablation == "no_codebook":
            scores_en = F.log_softmax(scores_ob, dim=1)
            return scores_en, F.nll_loss(scores_en, label)

        scores_en = F.log_softmax(scores_ob, dim=1)
        loss = F.nll_loss(scores_en, label) + 0.1 * vq_loss + extra_loss

        return scores_en, loss
