from typing import List


class Action:
    _value: List[int]

    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value
