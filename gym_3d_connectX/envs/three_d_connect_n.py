import copy
import gym
import torch
import pandas as pd
import plotly.express as px

from gym_3d_connectX.envs.utility import UtilClass


class AnyNumberInARow3dEnv(gym.Env):
    """
    the extended implementation of Five in a Row (Any Number in a Row) environment in manner of OpenAI gym Five
    in a Row is one of the most famous traditional board games in Japan. The rule of this game is simple.
    1. Two players puts the Go pieces (black & white stones) alternately on an empty intersection
    2. The winner is the first player to form an unbroken chain of five stones horizontally, vertically,
    or diagonally We extended this game to in two ways.
    First, we added another dimention to the board (2D to 3D).
    Second, we extended the required number for winning (five) to hyperparameter,
    which means programmers can set that number at their will.
    So, we can call the extended style game "Any Number in a Row"

    This class gives "Any Number in a Row" environment following OpenAI Gym interface.

    Attributes
    ----------
    num_grid : int
        The number of intersections in a board.
    step_number : int
        The number of current turns.
    obs_history : list[list[list[list[int]]]]
        An array containing the observations of one episode.
    board : list[list[list[int]]]
        A three-dimensional array representing the current state.
    action_space : gym.spaces
        Define an NxN discrete action space.
    observation_space : gym.spaces
        Define an N x N x N discrete space with three values (-1, 0, 1).
    player : int
        Define which is the first player.
    utils : UtilClass
        Utility class implemented in utility.py.
    """

    def __init__(self, num_grid=4, num_win_seq=4, win_reward=10, draw_penalty=5, lose_penalty=10,
                 could_locate_reward=0.1, couldnt_locate_penalty=0.1, time_penalty=0.1, first_player=1):
        """
        Parameters
        ----------
        num_grid : int
            Length of a side.
        num_win_seq : int
            The number of sequence necessary for winning.
        win_reward : float
            The reward agent gets when win the game.
        draw_penalty : float
            The penalty agent gets when it draw the game.
        lose_penalty : float
            The penalty agent gets when it lose the game.
        could_locate_reward : float
            The additional reward for agent being able to put the stone.
        couldnt_locate_penalty : float
            The penalty agent gets when it choose the location where the stone cannot be placed.
        time_penalty : float
            The penalty agents gets along with timesteps.
        first_player : int
            Define which is the first player.
        """
        super().__init__()

        self.num_grid = num_grid
        self.step_number = 0
        self.obs_history = []
        self.board = []

        # 重力がある設定（高さ方向は石を置く位置を指定できない）ので、N×Nの離散空間。
        self.action_space = gym.spaces.Discrete(self.num_grid * self.num_grid)
        # 自分の色の石が置かれている状態、石の置かれていない状態、相手プレイヤーの石が置かれている状態の3つをそれぞれ-1,0,1の値で表す。
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_grid, self.num_grid, self.num_grid))

        self.player = first_player

        # 継承すると必要以上に依存してしまうため、避けている。
        self.utils = UtilClass(
            num_grid=num_grid,
            num_win_seq=num_win_seq,
            win_reward=win_reward,
            draw_penalty=draw_penalty,
            lose_penalty=lose_penalty,
            could_locate_reward=could_locate_reward,
            couldnt_locate_penalty=couldnt_locate_penalty,
            time_penalty=time_penalty
        )

        # 環境の初期化
        self.reset()

    def reset(self):
        """
        Reset the board to the initial state.

        Returns
        -------
        reset : torch.Tensor
            The initial board tensor filled with 0 (0 means empty, 1 or -1 means the stone is put).
        """
        self.step_number = 0
        self.obs_history = []
        self.board = [[[0] * self.num_grid for _ in range(self.num_grid)] for _ in range(self.num_grid)]
        return torch.tensor(self.board).float()

    def step(self, action):
        """
        OpenAI gym style step function

        Receive the action and make transition.

        Parameter
        ---------
        action : int
            Elected aciton number (range from 0 to self.num_grid**2).

        Returns
        -------
        obs : torch.Tensor
            The observation agents get after the transition.
        reward : float
            The total reward agents get through the transition.
        done : bool
            The flag of whether the episode has finished or not.
        info : dict
            A dictionary containing the following information.
        """
        # 1~self.num_grid**2 の数値で表される action を、「升目のどの位置か」と言う情報に変換
        action = self.utils.base_change(action, self.num_grid).zfill(2)

        # 上記変換後、 action は縦横何マス目かを表す2文字の文字列（ex. '13'なら横2マス目、縦4マス目）になっているので、
        # それぞれの次元について位置を整数型にして取得。
        W = int(action[0])
        D = int(action[1])

        # 石の配置のダイナミクスを司る部分。石を配置し、次状態を返す。また、石を置ける場所を選択したかどうかに基づいて、追加情報（及び調整報酬）を返す。
        fixment_reward, self.board, is_couldnt_locate = self.utils.resolve_placing(
            wide=W,
            depth=D,
            player_number=self.player,
            board=self.board
        )

        # 現在のボードの状態から、ゲーム終了判定をし、（もし終了している場合）試合結果に応じた報酬および勝者情報を返す。
        done, reward, winner = self.utils.is_game_end(
            player_number=self.player,
            board=self.board
        )

        # このステップがどちらのプレーヤーによってなされたか、勝者はどちらか、このステップでプレーヤーは石の置ける場所を選択したか、の3つの情報を格納した辞書。
        info = {"turn": self.player, "winner": winner, "is_couldnt_locate": is_couldnt_locate}

        # プレーヤーの交代(置けない場所に置いていた場合は、プレーヤーは交代しない)
        if not is_couldnt_locate:
            # 両方のプレイヤーが一度置いてから格納する.(プレイヤーが表示されないバグが発生するため)
            if self.step_number > 0:
                # 参照渡しだと格納したデータが更新されてしまうため, アドレスを変更させる.
                self.obs_history.append(copy.deepcopy(self.board))
            self.step_number += 1

            self.player *= -1

        return torch.tensor(self.board).float(), reward + fixment_reward, done, info

    def render(self, mode="print"):
        """
        The function to draw the observation result of one step according to mode.

        Parameters
        ----------
        mode : str
            The flag to determine the content to be drawn.
        """

        if mode == "print":
            for i, square in enumerate(self.board):
                print(f"{i}F")
                for line in square:
                    print(line)

        elif mode == "plot":
            data = pd.DataFrame(index=[], columns=["W", "D", "H", "Player"])
            index = 0
            for i in range(self.num_grid):
                for j in range(self.num_grid):
                    for k in range(self.num_grid):
                        data.loc[index] = ([j, k, i, self.board[i][j][k]])
                        index += 1

            range_list = [-0.4, self.num_grid - 0.6]
            fig = px.scatter_3d(data, x="W", y="D", z="H", color="Player",
                                range_x=range_list, range_y=range_list, range_z=range_list,
                                color_discrete_map={0: "rgba(0,0,0,0)", 1: "red", -1: "blue"},
                                opacity=0.95, width=854, height=480)
            fig.show()

    def animation(self):
        """
        The function to draw the result of one episode of observation.
        """
        data = pd.DataFrame(index=[], columns=["W", "D", "H", "Player", "frame"])
        index = 0
        for frame, obs_history in enumerate(self.obs_history):
            for i in range(self.num_grid):
                for j in range(self.num_grid):
                    for k in range(self.num_grid):
                        data.loc[index] = ([j, k, i, obs_history[i][j][k], frame])
                        index += 1

        range_list = [-0.4, self.num_grid - 0.6]
        fig = px.scatter_3d(data, x="W", y="D", z="H", color="Player",
                            animation_frame="frame",
                            color_discrete_map={0: "rgba(0,0,0,0)", -1: "red", 1: "blue"},
                            range_color=[-1, 1],
                            range_x=range_list, range_y=range_list, range_z=range_list,
                            opacity=0.95, width=854, height=480)
        fig.show()


class Conv3dObsWrapper(gym.ObservationWrapper):
    """
    Define the wrapper class when using Conv3d.

    Attribute
    ---------
    observation_space : gym.space
        Define an N x N x N discrete space with three values (-1, 0, 1).
    """
    def __init__(self, env):
        """
        Parameter
        ---------
        env : gym.Env
            The gym environment.
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1, self.num_grid, self.num_grid, self.num_grid))

    def observation(self, obs):
        """
        Returns a new tensor with a dimension of size one inserted at the specified position.

        Parameter
        ---------
        obs : torch.Tensor
            The observation agents get after the transition.

        Return
        ------
        obs : torch.Tensor
            The observation agents get after the transition.
        """
        return torch.unsqueeze(obs, 0)
