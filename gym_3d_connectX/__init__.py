from gym.envs.registration import register

register(
    id='3d-connectX-v0',
    entry_point='gym_3d_connectX.env:AnyNumberInARow3dEnv'
)
