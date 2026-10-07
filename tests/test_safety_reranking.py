from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from evaluate_safety_reranking import (
    minmax,
    paired_bootstrap,
    ranking_metrics,
    select_penalty,
)


class SafetyRerankingTests(unittest.TestCase):
    def test_minmax_handles_constant_scores(self) -> None:
        result = minmax(np.asarray([2.0, 2.0, 2.0]))
        np.testing.assert_allclose(result, np.asarray([0.5, 0.5, 0.5]))

    def test_ranking_metrics_rewards_earlier_positive(self) -> None:
        labels = np.asarray([1, 0, 0], dtype=np.int64)
        early = ranking_metrics(labels, np.asarray([0.9, 0.2, 0.1]), 3)
        late = ranking_metrics(labels, np.asarray([0.1, 0.9, 0.2]), 3)
        self.assertGreater(early["ap"], late["ap"])
        self.assertGreater(early["mrr"], late["mrr"])
        self.assertEqual(early["recall@3"], 1.0)

    def test_penalty_selection_enforces_efficacy_constraint(self) -> None:
        frame = pd.DataFrame(
            {
                "lambda": [0.0, 0.5, 1.0],
                "efficacy_ap": [0.70, 0.695, 0.68],
                "top3_observed_risk_rate": [0.60, 0.50, 0.40],
            }
        )
        selected, annotated = select_penalty(frame, efficacy_tolerance=0.01)
        self.assertEqual(selected, 0.5)
        self.assertEqual(int(annotated["selected"].sum()), 1)

    def test_bootstrap_risk_reduction_has_positive_direction(self) -> None:
        frame = pd.DataFrame(
            {
                "fold": [1, 1, 1, 1],
                "symptom_index": [0, 0, 1, 1],
                "drug_index": [0, 0, 0, 0],
                "lambda": [0.0, 1.0, 0.0, 1.0],
                "efficacy_ap": [0.8, 0.8, 0.7, 0.7],
                "safe_ap": [0.8, 0.8, 0.7, 0.7],
                "top3_observed_risk_rate": [0.6, 0.4, 0.5, 0.3],
                "risky_efficacy_exposure@3": [0.5, 0.4, 0.4, 0.3],
            }
        )
        result = paired_bootstrap(frame, selected_lambda=1.0, samples=50)
        risk_row = result[
            result["comparison"] == "top3_observed_risk_rate_reduction"
        ].iloc[0]
        self.assertAlmostEqual(float(risk_row["point_estimate"]), 0.2)


if __name__ == "__main__":
    unittest.main()
