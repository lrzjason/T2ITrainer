from abc import ABC, abstractmethod

class ModelWrapper(ABC):
    def __init__(self):
        pass
    
    # @abstractmethod
    # def create(self):
    #     pass

    @abstractmethod
    def execute(self,model,image=None):
        pass
    