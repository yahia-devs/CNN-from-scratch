import numpy as np 




class Dense :
    def __init__ (self , n_input , n_output):
        self.W = np.random(n_input,n_output)
        self.b = np.zeros((1,n_output))
        self.dW = None
        self.db = None
        self.input = None
        
    def forward (self , x ):
        self.input = x 
        return x@self.W + self.b
    
    def backward (self , d_out):
        batch_size = d_out.shape[0]
        self.dW = (self.input.T @ d_out)/batch_size
        self.db = np.mean(d_out,axis=0)
        
        return d_out@self.W.T
    
    def update(self, eta):
        self.W -= eta * self.dW
        self.b -= eta * self.db