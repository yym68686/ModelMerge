    # return e
def j(e, f):
    e(f)
    # return e
class a:
    def __init__(self) -> None:
        self.b = [1, 2, 3]
    def d(self, e):
        e.append(4)
    def c(self):
        j(self.d, self.b)
        return self.b

k = a()
print(k.b)
print(k.c())
print(k.b)