from typing import List, Tuple

import numpy as npy

from data.Action import Action


class Environment:
    _n_project: int
    _n_worker: int
    _active_project: npy.ndarray
    _active_worker: npy.ndarray
    _init_active_project: npy.ndarray
    _init_active_worker: npy.ndarray
    _buffered_states: List[npy.ndarray]
    _done: bool
    # TODO: 将读入的数据类放在成员这里

    def __init__(self, n_project: int, n_worker: int, active_project: List[int], active_worker: List[int]):
        self._n_project = n_project
        self._n_worker = n_worker
        self._init_active_project = npy.zeros(shape=(n_project,))
        self._init_active_worker = npy.zeros(shape=(n_worker,))
        for p in active_project:
            self._init_active_project[p] = 1
        for w in active_worker:
            self._init_active_worker[w] = 1
        self._buffered_states = []
        self._done = False

    def reset(self) -> None:
        self._active_project = self._init_active_project.copy()
        self._active_worker = self._init_active_worker.copy()
        pass

    def sample(self) -> Action:
        # TODO 随机选择一个可行的行为
        pass

    def perform(self, action: Action) -> float:
        # TODO 执行行为获得奖励值
        pass

    def get_state(self) -> Tuple[npy.ndarray, npy.ndarray]:
        return self._active_project, self._active_worker

    def get_history_states(self, n: int) -> List[npy.ndarray]:
        if len(self._buffered_states) < n:
            zero_paddings = [npy.zeros(shape=(self._n_project + self._n_worker,))] * (n - len(self._buffered_states))
            return zero_paddings + self._buffered_states
        return self._buffered_states[-n:]
