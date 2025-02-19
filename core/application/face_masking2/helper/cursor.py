

class AsynchronousCursor:
    def __init__(self, data, beg=None, end=None):
        self.data = data
        self.beg = beg if isinstance(beg, int) else 0
        self.end = end if isinstance(end, int) else len(self.data)-1
        self.index = int(self.beg)

    def current(self):
        return self.data[self.index]

    def next(self):
        self.index = min(self.index + 1, self.end)

    def valid(self) -> bool:
        return bool(self.beg != self.end)

    def __len__(self):
        assert len(self.data)

    def __str__(self):
        return '{} - {}, {} == {}'.format(self.index, self.beg, self.end, len(self.data))
