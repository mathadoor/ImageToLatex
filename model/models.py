# Define the model architectures here

class baseWatcher:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self):
        pass

    def test(self):
        pass

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

class baseParser:
    def __init__(self, config):
        self.config = config

    def parse(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
class VanilleCNNWatcher(baseWatcher):
    def __init__(self, model, config):
        super().__init__(model, config)

    def train(self):
        pass

    def test(self):
        pass

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

class VanilleCNNParser(baseParser):
    def __init__(self, config):
        super().__init__(config)

    def parse(self):
        pass

    def save(self):
        pass

    def load(self):
        pass