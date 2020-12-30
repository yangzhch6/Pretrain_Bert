data = [1,2,3,4]
print(data)
class test():
    def __init__(self, data):
        self.data = data
    def change(self, d):
        self.data[0] = d
        return self.data

t = test(data)
new_d = t.change(-100)
print(new_d)
print(data)