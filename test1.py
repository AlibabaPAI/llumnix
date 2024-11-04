class Test:
    def __init__(self, *args,value):
        self.llargs=args
        self.value = value
    
    def process(self):
        sink(self.name)


def sink(a,b,c):
    print(b)

a=Test(1,2,3,value=4)
sink(*(a.llargs))
print(a.value)

