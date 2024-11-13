from test1 import Test
class tyest(Test):
    def process(self):
        def sink(value):
            print(2)
        sink(self.name)
        super().process()
        # sink(self.name) 
t = tyest("test", "value")
t.process()
