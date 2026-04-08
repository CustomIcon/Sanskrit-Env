import unittest

from server.environment import SanskritEnvironment


class EnvironmentScoringTests(unittest.TestCase):
    def setUp(self):
        self.env = SanskritEnvironment()

    def test_single_step_scores_are_shaped_in_environment(self):
        self.assertEqual(self.env._normalize_score(0, 0, 1, "glossary_anchoring"), 0.50)
        self.assertEqual(self.env._normalize_score(1, 0, 1, "glossary_anchoring"), 0.95)
        self.assertEqual(self.env._normalize_score(0, 1, 1, "glossary_anchoring"), 0.68)

    def test_coherence_final_scores_are_shaped_in_environment(self):
        raw_score = self.env._coherence_grader.compute_episode_score(
            final_reward=self.env._coherence_grader.MAIN_CORRECT,
            checkpoint_rewards=[
                self.env._coherence_grader.CHECKPOINT_CORRECT,
                self.env._coherence_grader.CHECKPOINT_CORRECT,
                self.env._coherence_grader.CHECKPOINT_CORRECT,
            ],
        )

        self.assertEqual(raw_score, 1.0)
        self.assertEqual(self.env._shape_episode_score(raw_score), 0.95)
        self.assertEqual(self.env._shape_episode_score(0.0), 0.50)


if __name__ == "__main__":
    unittest.main()