import numpy as np


class Convolution:


    def __init__ (self , num_filter, nb_canaux , n ):
        self.num_filter = num_filter
        self.nb_canaux = nb_canaux
        self.n = n
        self.weights = np.random.randn(self.num_filter,self.nb_canaux , self.n , self.n )
        self.bias = np.zeros(self.num_filter)



    def frward (self , image ):
        img = image
        h = img.shape[1]
        l = img.shape[2]

        h_n = h - self.n + 1
        l_n = l - self.n + 1
        res = np.zeros((self.num_filter,h_n,l_n))
        for f in range(self.num_filter):
            for i in range(h_n):
                for j in range(l_n ):
                    tmp = image[:, i:i+self.n ,j: j+ self.n]
                    res[f,i,j] =  np.sum(tmp * self.weights[f] )+ self.bias[f]
        return res





