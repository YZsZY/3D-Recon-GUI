class father:
    def __init__(self):
        self.name = "father"

    class son:
        def __init__(self):
            self.name = "son"
            self.print_name()


        def print_name(self):
            print(self.name)


father().son()
