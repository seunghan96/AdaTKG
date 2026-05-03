"""
AdaTKG online-memory adapters.

This module exposes the three update operators studied in the paper:
  - OnlineAdapter   : GRU cell                    (AdaTKG-GRU,      enhancement="meta")
  - EMAAdapter      : learnable EMA               (AdaTKG-EMA,      enhancement="ema"; default)
  - AttentionAdapter: cross-attention over buffer (AdaTKG-CrossAtt, enhancement="attention")

EMAAdapter additionally supports three ablation modes used in Section 4
of the paper:
  - decay_mode    in {"shared", "perentity", "perdim"}
  - gate_mode     in {"adaptive", "constant"}  (the constant-gate baseline)

The fusion rule (Eq. (7) of the paper) is shared across all three operators:
    z_e = (1 - g_e) * static_embs + g_e * memory_e
with g_e = 0 forced when the entity has no interaction observed yet
(the cold-start zero-mask underlying Corollary 1).
"""

import torch
import torch.nn as nn


# ================================================================
# AdaTKG-GRU (enhancement="meta")
# ================================================================
class OnlineAdapter(nn.Module):
    """GRU-based online adaptation for emerging entities.

    Each new interaction updates an entity-level hidden state via a GRU cell;
    a learned gate blends the adaptive state with the static embedding.
    Memory persists across batches within an epoch and is reset between epochs.
    """

    def __init__(self, embed_dim, num_entities):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_entities = num_entities

        self.gru_cell = nn.GRUCell(embed_dim, embed_dim)
        self.interaction_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.register_buffer("entity_memory", torch.zeros(num_entities, embed_dim))
        self.register_buffer("update_count", torch.zeros(num_entities))

    def reset_memory(self):
        self.entity_memory.zero_()
        self.update_count.zero_()

    @torch.no_grad()
    def update_entities(self, entity_ids, interaction_embs):
        unique_ids = entity_ids.unique()
        for eid in unique_ids:
            mask = entity_ids == eid
            avg_inter = interaction_embs[mask].mean(dim=0, keepdim=True)
            cur = self.entity_memory[eid : eid + 1]
            new = self.gru_cell(avg_inter, cur)
            self.entity_memory[eid] = new.squeeze(0)
            self.update_count[eid] += 1

    def get_adaptive_embedding(self, entity_ids, static_embs):
        mem = self.entity_memory[entity_ids]
        has_mem = (self.update_count[entity_ids] > 0).float().unsqueeze(-1)
        g = self.gate_net(torch.cat([static_embs, mem], dim=-1)) * has_mem
        return static_embs * (1 - g) + mem * g

    def forward(self, query_entities, static_embs, chain_embs=None, relation_embs=None):
        if chain_embs is not None and relation_embs is not None:
            inter = self.interaction_encoder(
                torch.cat([chain_embs, relation_embs], dim=-1)
            )
            self.update_entities(query_entities, inter.detach())
        return self.get_adaptive_embedding(query_entities, static_embs)


# ================================================================
# AdaTKG-EMA (enhancement="ema") — default operator
# ================================================================
class EMAAdapter(nn.Module):
    """Exponential moving average — learned decay, no neural update rule.

    `decay_mode` controls the parameterization of the EMA decay:
      - "shared"    : a single learned scalar  alpha = sigma(rho)            (default)
      - "perentity" : per-entity scalar        alpha_e = sigma(rho_e)        (rho shape: [E])
      - "perdim"    : per-dimension vector     alpha = sigma(rho), rho in R^d
    `gate_mode` controls the fusion gate:
      - "adaptive"  : learned gate net              (default)
      - "constant"  : g = const_gate, zero-masked   (the constant-gate ablation)
    """

    def __init__(self, embed_dim, num_entities, decay_mode="shared",
                 gate_mode="adaptive", const_gate=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_entities = num_entities
        self.decay_mode = decay_mode
        self.gate_mode = gate_mode
        self.const_gate = float(const_gate)
        if decay_mode == "shared":
            self.raw_decay = nn.Parameter(torch.tensor(2.0))
        elif decay_mode == "perentity":
            self.raw_decay = nn.Parameter(torch.full((num_entities,), 2.0))
        elif decay_mode == "perdim":
            self.raw_decay = nn.Parameter(torch.full((embed_dim,), 2.0))
        else:
            raise ValueError(f"unknown decay_mode: {decay_mode}")
        self.interaction_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gate_net = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        self.register_buffer("entity_memory", torch.zeros(num_entities, embed_dim))
        self.register_buffer("update_count", torch.zeros(num_entities))

    @property
    def decay(self):
        return torch.sigmoid(self.raw_decay)

    def reset_memory(self):
        self.entity_memory.zero_()
        self.update_count.zero_()

    @torch.no_grad()
    def update_entities(self, entity_ids, interaction_embs):
        d_all = self.decay.detach()
        unique_ids = entity_ids.unique()
        for eid in unique_ids:
            mask = entity_ids == eid
            avg = interaction_embs[mask].mean(dim=0)
            if self.decay_mode == "shared":
                d = d_all
            elif self.decay_mode == "perentity":
                d = d_all[eid]
            else:  # perdim
                d = d_all
            self.entity_memory[eid] = d * self.entity_memory[eid] + (1 - d) * avg
            self.update_count[eid] += 1

    def get_adaptive_embedding(self, entity_ids, static_embs):
        mem = self.entity_memory[entity_ids]
        has_mem = (self.update_count[entity_ids] > 0).float().unsqueeze(-1)
        if self.gate_mode == "constant":
            g = torch.full_like(static_embs, self.const_gate) * has_mem
        else:
            g = self.gate_net(torch.cat([static_embs, mem], dim=-1)) * has_mem
        return static_embs * (1 - g) + mem * g

    def forward(self, query_entities, static_embs, chain_embs=None, relation_embs=None):
        if chain_embs is not None and relation_embs is not None:
            inter = self.interaction_encoder(torch.cat([chain_embs, relation_embs], dim=-1))
            self.update_entities(query_entities, inter.detach())
        return self.get_adaptive_embedding(query_entities, static_embs)


# ================================================================
# AdaTKG-CrossAtt (enhancement="attention")
# ================================================================
class AttentionAdapter(nn.Module):
    """Store the most recent K interaction signals per entity in a FIFO buffer
    and read out the memory state by cross-attending the query to the buffer."""

    def __init__(self, embed_dim, num_entities, buffer_size=16, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_entities = num_entities
        self.buffer_size = buffer_size
        self.interaction_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.gate_net = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        self.register_buffer("buffer", torch.zeros(num_entities, buffer_size, embed_dim))
        self.register_buffer("buffer_len", torch.zeros(num_entities, dtype=torch.long))

    def reset_memory(self):
        self.buffer.zero_()
        self.buffer_len.zero_()

    @torch.no_grad()
    def update_entities(self, entity_ids, interaction_embs):
        for i, eid in enumerate(entity_ids):
            eid = eid.item()
            pos = self.buffer_len[eid] % self.buffer_size
            self.buffer[eid, pos] = interaction_embs[i]
            self.buffer_len[eid] += 1

    def get_adaptive_embedding(self, entity_ids, static_embs, query_embs):
        B = len(entity_ids)
        buf = self.buffer[entity_ids]
        lengths = self.buffer_len[entity_ids].clamp(max=self.buffer_size)
        has_mem = (lengths > 0).float().unsqueeze(-1)
        mask = (
            torch.arange(self.buffer_size, device=buf.device)
            .unsqueeze(0)
            .expand(B, -1) >= lengths.unsqueeze(-1)
        )
        q = query_embs.unsqueeze(1)
        mem, _ = self.cross_attn(q, buf, buf, key_padding_mask=mask)
        mem = mem.squeeze(1)
        g = self.gate_net(torch.cat([static_embs, mem], dim=-1)) * has_mem
        return static_embs * (1 - g) + mem * g

    def forward(self, query_entities, static_embs,
                chain_embs=None, relation_embs=None, query_embs=None):
        if chain_embs is not None and relation_embs is not None:
            inter = self.interaction_encoder(torch.cat([chain_embs, relation_embs], dim=-1))
            self.update_entities(query_entities, inter.detach())
        if query_embs is None:
            query_embs = static_embs
        return self.get_adaptive_embedding(query_entities, static_embs, query_embs)
