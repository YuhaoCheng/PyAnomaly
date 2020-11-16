class Test(object):
    def __init__(self) -> None:
        self.a1 = 1
        self.__setattr__("b", 2)
    def hh(self):
        print(self.b)

if __name__ == '__main__':
    temp = Test()
    test = {'1': 123, '2':234}
    for item in test.items():
        import ipdb; ipdb.set_trace()