from abc import ABC, abstractmethod
from typing import Dict


class Mechanism(ABC):

    @abstractmethod
    def abduct(self, cond: Dict) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def predict(self, noise: Dict, interv: Dict) -> Dict:
        raise NotImplementedError