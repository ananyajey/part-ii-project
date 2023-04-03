import numpy as np
from .layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        out = np.dot(self.weights, self.input) + self.bias
        return out
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T) # dE/dW
        bias_gradient = output_gradient # dE/dB

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        input_gradient = np.dot(self.weights.T, output_gradient) # dE/dX
        
        return input_gradient
    
    




'''
import numpy as np

class ConvolutionalLayer:
    def __init__(self, kernel_count, kernel_size):
        self.kernel_count = kernel_count
        self.kernel_size = kernel_size

        self.kernels = np.random.randn(kernel_count, kernel_size, kernel_size) / (kernel_size ** 2)


    def patches_generator(self, image):
        image_height, image_width = image.shape
        self.image = image

        for h in range(image_height - self.kernel_size + 1):
            for w in range(image_width - self.kernel_size + 1):
                patch = image[h:(h + self.kernel_size), w:(w + self.kernel_size)]

                yield patch, h, w


    def forward_propagation(self, image):
        image_height, image_width = image.shape
        output = np.zeros((image_height - self.kernel_size + 1, image_width - self.kernel_size + 1, self.kernel_count))

        for patch, height, width in self.patches_generator(image):
            output[height, width] = np.sum(patch * self.kernels, axis = (1, 2))
        
        return output


    def back_propagation(self, dEdY, alpha):
        output = np.zeors(self.kernels.shape)
        
        for patch, height, width, in self.patches_generator(self.image):
            for k in range(self.kernel_count):
                output[k] += patch * dEdY[height, width, k]

        self.kernels -= alpha*output

        return output



class MaxPoolingLayer:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def patches_generator(self, image):
        height, width = image.shape[0], image.shape[1]
        self.image = image

        for h in range(height):
            for w in range(width):
                a = self.kernel_size * h
                b = a + self.kernel_size
                c = self.kernel_size * w
                d = c + self.kernel_size

                patch = image[a:b, c:d]

                yield patch, h, w
    
    def forward_propagation(self, image):
        image_height, image_width, kernel_count = image.shape
        
        x = int(image_height/self.kernel_size)
        y = int(image_width/self.kernel_size)

        output = np.zeros((x, y, kernel_count))
        
        for patch, height, width in self.patches_generator(image):
            output[height, width] = np.amax(patch, axis=(0, 1))

        return output




    def back_propagation(self, dEdY):
        output = np.zeros(self.image.shape)

        for patch, height, width in self.patches_generator(self.image):
            image_height, image_width, kernel_count = patch.shape
            max_value = np.amax(patch, axis = (0, 1))

            for h in range(image_height):
                for w in range(image_width):
                    for k in range(kernel_count):
                        if patch[h, w, k] == max[k]:
                            a = height*self.kernel_size + h
                            b = width*self.kernel_size + w
                            output[a, b, k] = dEdY[height, width, k]
            
            return output
        


#class SoftmaxLayer:
        
        
    
'''
