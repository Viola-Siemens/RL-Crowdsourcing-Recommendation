class Action:
    _value: int

    def __init__(self, value: int):
        self._value = value

    def get(self) -> int:
        return self._value
