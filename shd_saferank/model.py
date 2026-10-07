from __future__ import annotations

import math

import torch
from torch import nn


class MultiTaskSafeRank(nn.Module):
    def __init__(
        self,
        n_herbs: int,
        n_symptoms: int,
        n_drugs: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        hdi_feature_mode: str = "both",
    ) -> None:
        super().__init__()
        if hdi_feature_mode not in {"both", "herb_only", "drug_only"}:
            raise ValueError(f"Unknown HDI feature mode: {hdi_feature_mode}")
        self.embedding_dim = embedding_dim
        self.hdi_feature_mode = hdi_feature_mode
        self.herb_embedding = nn.Embedding(n_herbs, embedding_dim)
        self.symptom_embedding = nn.Embedding(n_symptoms, embedding_dim)
        self.drug_embedding = nn.Embedding(n_drugs, embedding_dim)
        self.herb_bias = nn.Embedding(n_herbs, 1)
        self.symptom_bias = nn.Embedding(n_symptoms, 1)
        self.risk_head = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.herb_embedding.weight)
        nn.init.xavier_uniform_(self.symptom_embedding.weight)
        nn.init.xavier_uniform_(self.drug_embedding.weight)
        nn.init.zeros_(self.herb_bias.weight)
        nn.init.zeros_(self.symptom_bias.weight)
        for layer in self.risk_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def score_herb_symptom(
        self, herb_indices: torch.Tensor, symptom_indices: torch.Tensor
    ) -> torch.Tensor:
        herb = self.herb_embedding(herb_indices)
        symptom = self.symptom_embedding(symptom_indices)
        score = (herb * symptom).sum(dim=-1) / math.sqrt(self.embedding_dim)
        return (
            score
            + self.herb_bias(herb_indices).squeeze(-1)
            + self.symptom_bias(symptom_indices).squeeze(-1)
        )

    def score_herb_drug_risk(
        self, herb_indices: torch.Tensor, drug_indices: torch.Tensor
    ) -> torch.Tensor:
        herb = self.herb_embedding(herb_indices)
        drug = self.drug_embedding(drug_indices)
        if self.hdi_feature_mode == "herb_only":
            drug = torch.zeros_like(drug)
        elif self.hdi_feature_mode == "drug_only":
            herb = torch.zeros_like(herb)
        features = torch.cat(
            [herb, drug, herb * drug, torch.abs(herb - drug)], dim=-1
        )
        return self.risk_head(features).squeeze(-1)


class InductiveMultiTaskSafeRank(nn.Module):
    """Shared-private multi-task model driven by label-free entity features."""

    def __init__(
        self,
        herb_target_features: torch.Tensor,
        herb_component_features: torch.Tensor,
        herb_feature_mask: torch.Tensor,
        symptom_features: torch.Tensor,
        drug_features: torch.Tensor,
        drug_feature_mask: torch.Tensor,
        drug_target_features: torch.Tensor,
        drug_target_feature_mask: torch.Tensor,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        herb_architecture: str = "shared_private",
    ) -> None:
        super().__init__()
        if herb_architecture not in {"shared_private", "shared_only", "private_only"}:
            raise ValueError(f"Unknown herb architecture: {herb_architecture}")
        self.embedding_dim = embedding_dim
        self.herb_architecture = herb_architecture
        self.register_buffer("herb_target_features", herb_target_features.float())
        self.register_buffer(
            "herb_component_features", herb_component_features.float()
        )
        self.register_buffer("herb_feature_mask", herb_feature_mask.float())
        self.register_buffer("symptom_features", symptom_features.float())
        self.register_buffer("drug_features", drug_features.float())
        self.register_buffer("drug_feature_mask", drug_feature_mask.float())
        self.register_buffer("drug_target_features", drug_target_features.float())
        self.register_buffer(
            "drug_target_feature_mask", drug_target_feature_mask.float()
        )

        herb_input_dim = (
            herb_target_features.shape[1]
            + herb_component_features.shape[1]
            + herb_feature_mask.shape[1]
        )
        self.shared_herb_encoder = self._encoder(
            int(herb_input_dim), hidden_dim, embedding_dim, dropout
        )
        self.efficacy_herb_encoder = self._encoder(
            int(herb_input_dim), hidden_dim, embedding_dim, dropout
        )
        self.risk_herb_encoder = self._encoder(
            int(herb_input_dim), hidden_dim, embedding_dim, dropout
        )
        self.symptom_encoder = self._encoder(
            int(symptom_features.shape[1]), hidden_dim, embedding_dim, dropout
        )
        self.drug_encoder = self._encoder(
            int(drug_features.shape[1] + drug_target_features.shape[1] + 2),
            hidden_dim,
            embedding_dim,
            dropout,
        )
        self.efficacy_gate = nn.Linear(embedding_dim * 2, embedding_dim)
        self.risk_gate = nn.Linear(embedding_dim * 2, embedding_dim)
        self.symptom_bias = nn.Sequential(
            nn.Linear(int(symptom_features.shape[1]), hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    @staticmethod
    def _encoder(
        input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _herb_input(self, herb_indices: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.herb_target_features[herb_indices],
                self.herb_component_features[herb_indices],
                self.herb_feature_mask[herb_indices],
            ],
            dim=-1,
        )

    @staticmethod
    def _gated_mix(
        shared: torch.Tensor, private: torch.Tensor, gate_layer: nn.Linear
    ) -> torch.Tensor:
        gate = torch.sigmoid(gate_layer(torch.cat([shared, private], dim=-1)))
        return gate * shared + (1.0 - gate) * private

    def efficacy_herb_representation(
        self, herb_indices: torch.Tensor
    ) -> torch.Tensor:
        inputs = self._herb_input(herb_indices)
        shared = self.shared_herb_encoder(inputs)
        private = self.efficacy_herb_encoder(inputs)
        if self.herb_architecture == "shared_only":
            return shared
        if self.herb_architecture == "private_only":
            return private
        return self._gated_mix(shared, private, self.efficacy_gate)

    def risk_herb_representation(self, herb_indices: torch.Tensor) -> torch.Tensor:
        inputs = self._herb_input(herb_indices)
        shared = self.shared_herb_encoder(inputs)
        private = self.risk_herb_encoder(inputs)
        if self.herb_architecture == "shared_only":
            return shared
        if self.herb_architecture == "private_only":
            return private
        return self._gated_mix(shared, private, self.risk_gate)

    def score_herb_symptom(
        self, herb_indices: torch.Tensor, symptom_indices: torch.Tensor
    ) -> torch.Tensor:
        herb = self.efficacy_herb_representation(herb_indices)
        raw_symptom = self.symptom_features[symptom_indices]
        symptom = self.symptom_encoder(raw_symptom)
        score = (herb * symptom).sum(dim=-1) / math.sqrt(self.embedding_dim)
        return score + self.symptom_bias(raw_symptom).squeeze(-1)

    def score_herb_drug_risk(
        self, herb_indices: torch.Tensor, drug_indices: torch.Tensor
    ) -> torch.Tensor:
        herb = self.risk_herb_representation(herb_indices)
        drug_input = torch.cat(
            [
                self.drug_features[drug_indices],
                self.drug_feature_mask[drug_indices].unsqueeze(-1),
                self.drug_target_features[drug_indices],
                self.drug_target_feature_mask[drug_indices].unsqueeze(-1),
            ],
            dim=-1,
        )
        drug = self.drug_encoder(drug_input)
        features = torch.cat(
            [herb, drug, herb * drug, torch.abs(herb - drug)], dim=-1
        )
        return self.risk_head(features).squeeze(-1)
