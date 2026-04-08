import unittest

from models import ManuscriptAction
from server.environment import SanskritEnvironment


class EnvironmentScoringTests(unittest.TestCase):
    def setUp(self):
        self.env = SanskritEnvironment()

    def test_reward_signal_is_shaped_in_environment(self):
        self.assertEqual(self.env._shape_reward_signal(0.0), 0.50)
        self.assertEqual(self.env._shape_reward_signal(1.0), 0.95)
        self.assertEqual(self.env._shape_reward_signal(0.4), 0.68)

    def test_single_step_task_emits_non_binary_reward(self):
        episode = self.env._task1_data["episodes"][0]
        self.env.reset(task_id="glossary_anchoring", episode_id=episode["id"])

        observation = self.env.step(ManuscriptAction(selected_option=episode["correct_answer"]))

        self.assertEqual(observation.step_reward, 0.95)
        self.assertEqual(observation.reward, 0.95)
        self.assertEqual(observation.cumulative_score, 0.95)

    def test_coherence_step_and_final_scores_are_shaped_in_environment(self):
        episode = self.env._task3_data["episodes"][0]
        observation = self.env.reset(task_id="referential_coherence", episode_id=episode["id"])

        for checkpoint in episode.get("consistency_checkpoints", []):
            selected_option = next(
                option
                for option in observation.candidate_options
                if option.startswith(checkpoint["answer"])
            )
            observation = self.env.step(ManuscriptAction(selected_option=selected_option))
            self.assertGreater(observation.step_reward, 0.0)
            self.assertLess(observation.step_reward, 1.0)

        observation = self.env.step(ManuscriptAction(selected_option=episode["correct_answer"]))

        self.assertGreater(observation.step_reward, 0.0)
        self.assertLess(observation.step_reward, 1.0)
        self.assertGreater(observation.reward, 0.0)
        self.assertLess(observation.reward, 1.0)

    def test_coherence_episode_credit_still_maps_to_reward_band(self):
        raw_score = self.env._coherence_grader.compute_episode_score(
            final_reward=self.env._coherence_grader.MAIN_CORRECT,
            checkpoint_rewards=[
                self.env._coherence_grader.CHECKPOINT_CORRECT,
                self.env._coherence_grader.CHECKPOINT_CORRECT,
                self.env._coherence_grader.CHECKPOINT_CORRECT,
            ],
        )

        self.assertEqual(raw_score, 1.0)
        self.assertEqual(self.env._shape_reward_signal(raw_score), 0.95)
        self.assertEqual(self.env._shape_reward_signal(0.0), 0.50)


if __name__ == "__main__":
    unittest.main()