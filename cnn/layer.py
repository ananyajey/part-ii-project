class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # take input and give output
        pass

    def backward(self, output_gradient, learning_rate):
        # take derivative of error with respect to putput
        # update trainable parameters
        # return derivative of error with respect to input
        pass