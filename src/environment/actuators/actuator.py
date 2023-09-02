from abc import ABCMeta, abstractmethod


class Actuator(metaclass=ABCMeta):
    @abstractmethod
    def operate(action):
        raise NotImplementedError
