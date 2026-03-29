
import numpy as np

class Pool :


    def __init__(self, n):
        self.n = n



    def forward(self,input):
        self.input = input

        num_filters , h , l = input.shape
        h_out = h // self.n
        l_out = l // self.n

        res = np.zeros((num_filters , h_out,l_out))
        for f in range (num_filters):
            for i in range(0,h,self.n):
                for j in range(0,l,self.n):
                    val = np.max(input[f, i:i+self.n, j:j+self.n])
                    res[f][i//self.n][j//self.n] = val
        return res





