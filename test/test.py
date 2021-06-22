import unittest

from gym_3d_connect4.envs import AnyNumberInARow3dEnv


class TestCombination(unittest.TestCase):
    def setUp(self):
        self.pattern = [
            {
                "actions": [0, 0, 1, 1, 2, 2, 4, 4, 0, 0, 1, 1, 2, 2, 0, 3],
                "rewards": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, -10, 100],
                "dones": [False, False, False, False, False, False, False, False, False, False, False, False, False,
                          False,
                          False, True],
                "players": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1],
                "winners": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                "couldnt_locates": [False, False, False, False, False, False, False, False, False, False, False, False,
                                    False, False, True, False],
                "num_grid": 4,
                "num_win_seq": 4
            },
            {
                "actions": [0, 0, 0, 0, 8, 1],
                "rewards": [10, 10, 10, -10, 10, 100],
                "dones": [False, False, False, False, False, True],
                "players": [1, -1, 1, -1, -1, 1],
                "winners": [0, 0, 0, 0, 0, 1],
                "couldnt_locates": [False, False, False, True, False, False],
                "num_grid": 3,
                "num_win_seq": 2
            }
        ]

    def tearDown(self):
        print("Your environment has passed the test!!!!")

    def test_pattern(self):
        for pattern in self.pattern:
            self.env = AnyNumberInARow3dEnv(num_grid=pattern["num_grid"], num_win_seq=pattern["num_win_seq"])
            self.env.reset()
            # 各種報酬・罰の設定。（テスト用なので値はわかりやすくしてある　）
            self.env.utils.win_reward = 100
            self.env.utils.draw_penalty = 50
            self.env.utils.lose_penalty = 100
            self.env.utils.could_locate_reward = 10
            self.env.utils.couldnt_locate_penalty = 10
            self.env.utils.time_penalty = 1
            # （のちの解説を分かりやすくするため、）初期プレーヤーは1固定
            self.env.player = 1

            for idx, action in enumerate(pattern["actions"]):
                obs, reward, done, info = self.env.step(action)

                self.assertEqual(pattern["rewards"][idx], reward)
                self.assertEqual(pattern["dones"][idx], done)
                self.assertEqual(pattern["players"][idx], info["turn"])
                self.assertEqual(pattern["winners"][idx], info["winner"])
                self.assertEqual(pattern["couldnt_locates"][idx], info["is_couldnt_locate"])


if __name__ == '__main__':
    unittest.main()
