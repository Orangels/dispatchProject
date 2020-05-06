class Person:
    def __init__(self):
        self.info = []
    def push(self, name, sex, age):
        self.info.append((name, sex, age))
    def show(self):
        print(self.info)