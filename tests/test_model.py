from __future__ import annotations

import unittest

import torch

from shd_saferank.model import InductiveMultiTaskSafeRank, MultiTaskSafeRank


class ModelTests(unittest.TestCase):
    def test_model_output_shapes(self) -> None:
        model = MultiTaskSafeRank(10, 5, 7, embedding_dim=8, hidden_dim=12)
        herbs = torch.tensor([0, 1, 2], dtype=torch.long)
        symptoms = torch.tensor([1, 2, 3], dtype=torch.long)
        drugs = torch.tensor([2, 3, 4], dtype=torch.long)
        self.assertEqual(model.score_herb_symptom(herbs, symptoms).shape, (3,))
        self.assertEqual(model.score_herb_drug_risk(herbs, drugs).shape, (3,))

    def test_single_side_hdi_controls_ignore_masked_entity(self) -> None:
        herbs = torch.tensor([0, 1], dtype=torch.long)
        drugs = torch.tensor([0, 1], dtype=torch.long)
        drug_only = MultiTaskSafeRank(
            3, 2, 3, embedding_dim=8, hidden_dim=12, hdi_feature_mode="drug_only"
        )
        herb_only = MultiTaskSafeRank(
            3, 2, 3, embedding_dim=8, hidden_dim=12, hdi_feature_mode="herb_only"
        )
        drug_only.eval()
        herb_only.eval()
        self.assertTrue(
            torch.allclose(
                drug_only.score_herb_drug_risk(herbs, drugs),
                drug_only.score_herb_drug_risk(herbs.flip(0), drugs),
            )
        )
        self.assertTrue(
            torch.allclose(
                herb_only.score_herb_drug_risk(herbs, drugs),
                herb_only.score_herb_drug_risk(herbs, drugs.flip(0)),
            )
        )

    def test_inductive_model_output_shapes(self) -> None:
        model = InductiveMultiTaskSafeRank(
            herb_target_features=torch.randn(10, 6),
            herb_component_features=torch.randn(10, 8),
            herb_feature_mask=torch.ones(10, 2),
            symptom_features=torch.randn(5, 6),
            drug_features=torch.randn(7, 8),
            drug_feature_mask=torch.ones(7),
            drug_target_features=torch.randn(7, 6),
            drug_target_feature_mask=torch.ones(7),
            embedding_dim=8,
            hidden_dim=12,
        )
        herbs = torch.tensor([0, 1, 2], dtype=torch.long)
        symptoms = torch.tensor([1, 2, 3], dtype=torch.long)
        drugs = torch.tensor([2, 3, 4], dtype=torch.long)
        self.assertEqual(model.score_herb_symptom(herbs, symptoms).shape, (3,))
        self.assertEqual(model.score_herb_drug_risk(herbs, drugs).shape, (3,))


if __name__ == "__main__":
    unittest.main()
