from __future__ import annotations

import unittest

import numpy as np

from shd_saferank.data import (
    known_herbs_by_symptom,
    sample_hs_negative_herbs,
)


class ProtocolTests(unittest.TestCase):
    def test_negative_sampler_excludes_all_known_positive_herbs(self) -> None:
        edges = np.asarray([[0, 1], [2, 1], [1, 0]], dtype=np.int64)
        known = known_herbs_by_symptom(edges)
        symptoms, _, negatives = sample_hs_negative_herbs(
            positive_edges=edges,
            known_positives=known,
            candidate_herbs=np.arange(5, dtype=np.int64),
            negatives_per_positive=4,
            rng=np.random.default_rng(0),
        )
        for symptom, negative in zip(symptoms, negatives):
            self.assertNotIn(int(negative), known[int(symptom)])

    def test_known_herb_sets_are_query_specific(self) -> None:
        edges = np.asarray([[0, 1], [2, 1], [1, 2]], dtype=np.int64)
        known = known_herbs_by_symptom(edges)
        self.assertEqual(known, {1: {0, 2}, 2: {1}})


if __name__ == "__main__":
    unittest.main()
