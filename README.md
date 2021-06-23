# 3d-connect4-gym

[![BuildStatus][build-status]][ci-server]
[![PackageVersion][pypi-version]][pypi-home]
[![PythonVersion][python-version]][python-home]
[![Stable][pypi-status]][pypi-home]
[![Format][pypi-format]][pypi-home]
[![License][pypi-license]](LICENSE)

[build-status]: https://travis-ci.org/youngeek-0410/3d-connectX-env.svg?branch=master
[ci-server]: https://travis-ci.org/youngeek-0410/3d-connectX-env
[pypi-version]: https://badge.fury.io/py/3d-connectX-env.svg
[pypi-license]: https://img.shields.io/pypi/l/3d-connectX-env.svg
[pypi-status]: https://img.shields.io/pypi/status/3d-connectX-env.svg
[pypi-format]: https://img.shields.io/pypi/format/3d-connectX-env.svg
[pypi-home]: https://badge.fury.io/py/3d-connectX-env
[python-version]: https://img.shields.io/pypi/pyversions/3d-connectX-env.svg
[python-home]: https://python.org

3D connect4 repository, developed for the [OpenAI Gym](https://github.com/openai/gym) format.

## Installation

The preferred installation of `3d-connect4-gym` is from `pip`:

```shell
pip install 3d-connect4-gym
```

## Usage

### Python

```python
from gym_3d_connect4.envs import AnyNumberInARow3dEnv
env = AnyNumberInARow3dEnv()
env.reset()

env.utils.win_reward = 100
env.utils.draw_penalty = 50
env.utils.lose_penalty = 100
env.utils.could_locate_reward = 10
env.utils.couldnt_locate_penalty = 10
env.utils.time_penalty = 1
env.player = 1
actions = [0, 0, 1, 1, 2, 2, 4, 4, 0, 0, 1, 1, 2, 2, 0, 3]

for action in actions:
    obs, reward, done, info = env.step(action)
    env.render(mode="plot")

```
