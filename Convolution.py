import numpy as np


class Convolution:


    def __init__ (self , num_filter, nb_canaux , n ):
        self.num_filter = num_filter
        self.nb_canaux = nb_canaux
        self.n = n
        self.weights = np.random.randn(self.num_filter,self.nb_canaux , self.n , self.n )
        self.bias = np.zeros(self.num_filter)



    def forward (self , image ):
        self.img = image
        h = self.img.shape[1]
        l = self.img.shape[2]

        h_n = h - self.n + 1
        l_n = l - self.n + 1
        res = np.zeros((self.num_filter,h_n,l_n))
        for f in range(self.num_filter):
            for i in range(h_n):
                for j in range(l_n ):
                    tmp = image[:, i:i+self.n ,j: j+ self.n]
                    res[f,i,j] =  np.sum(tmp * self.weights[f] )+ self.bias[f]
        return res



    def backward (self , d_out ):
        h = self.img.shape[1]
        l = self.img.shape[2]

        h_n = h - self.n + 1
        l_n = l - self.n + 1
        d_weights = np.zeros(self.weights.shape)
        d_biases = np.zeros(self.bias.shape)
        d_input = np.zeros(self.img.shape)
        for f in range(self.num_filter):
            for i in range(h_n):
                for j in range(l_n ):
                    tmp = self.img[:, i:i+self.n ,j: j+ self.n]
                    d_weights[f] += tmp * d_out[f,i,j]
                    d_input[:, i:i + self.n, j:j + self.n] += self.weights[f] * d_out[f, i, j]

            d_biases[f] = np.sum(d_out[f])

        self.d_weights = d_weights
        self.d_biases = d_biases

        return d_input



    def update(self, eta):
        self.weights -= eta * self.d_weights
        self.bias -= eta * self.d_biases















