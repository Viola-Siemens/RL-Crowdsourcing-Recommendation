import random
from typing import List

import numpy as npy

from data.Action import Action
from data.DataLoader import Data

alpha = 0.2


class Environment:
    _state: npy.ndarray
    _buffered_states: List[npy.ndarray]
    _done: bool
    _data: Data
    _index: int
    _reward_type: str  # 'w' for workers, 'r' for requesters (linear), 'rn1' for requesters (non-linear1),
    # 'rn2' for requesters (non-linear2)

    def __init__(self, reward_type: str = 'w'):
        self._data = Data()
        self._data.get_data()
        self._reward_type = reward_type

    def reset(self) -> None:
        self._done = False
        self._index = 0
        self._state = self._data.get_state_array(self._index)
        self._buffered_states = []
        pass

    def sample(self) -> Action:
        # 随机选择一个可行的行为
        # n个worker，则返回 [0, n-1] ∩ N
        return Action(random.randint(0, len(self._data.worker_quality) - 1))

    def perform(self, action: Action) -> float:
        # 执行行为获得奖励值
        worker_id = self._data.get_worker_id_by_index(action.get())
        project_id = self._data.get_project_id_by_index(self._index)
        ret = self._data.get_standard_reward(worker_id, project_id)
        if self._reward_type == 'r':
            ret = alpha * ret + (1.0 - alpha) * self._data.get_quality_reward(worker_id)
        elif self._reward_type == 'rn1':
            ret *= self._data.get_quality_reward(worker_id)
        elif self._reward_type == 'rn2':
            ret = 1.0 - ((1.0 - ret) * (1.0 - self._data.get_quality_reward(worker_id)))
        self._index += 1
        if self._index >= self._data.get_projects_length():
            self._done = True
        else:
            self._state = self._data.get_state_array(self._index)

        return ret

    def is_done(self) -> bool:
        return self._done

    def get_state(self) -> npy.ndarray:
        return self._state

    def get_history_states(self, n: int) -> List[npy.ndarray]:
        if len(self._buffered_states) < n:
            zero_paddings = [npy.zeros(shape=(self._data.n_state,))] * (n - len(self._buffered_states))
            return zero_paddings + self._buffered_states
        return self._buffered_states[-n:]

    def get_state_dim(self) -> int:
        return self._data.n_state

    def get_output_dim(self) -> int:
        return len(self._data.worker_quality)

    def set_reward_type(self, reward_type: str) -> None:
        self._reward_type = reward_type
