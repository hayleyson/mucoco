import unittest

from laser_edit.evaluation.metrics import perspective_toxicity_score


class PerspectiveSummaryScoreTest(unittest.TestCase):
    def test_uses_summary_score_not_first_span(self):
        response = {
            "attributeScores": {
                "TOXICITY": {
                    "spanScores": [{"score": {"value": 0.91}}],
                    "summaryScore": {"value": 0.12},
                }
            }
        }

        self.assertEqual(perspective_toxicity_score(response), 0.12)
