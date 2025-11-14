
class PartialPQ:
    def __init__(self, M, B):
        self.M = M
        self.B = B
        self.data = []

    def insert(self, key, value):
        self.data.append((value, key))
        self.data.sort()

    def batch_prepend(self, items):
        for k,v in items:
            self.data.append((v,k))
        self.data.sort()

    def pull(self):
        if not self.data:
            return [], self.B
        chunk = self.data[: self.M]
        self.data = self.data[self.M:]
        S = [k for v,k in chunk]
        bound = self.data[0][0] if self.data else self.B
        return S, bound
