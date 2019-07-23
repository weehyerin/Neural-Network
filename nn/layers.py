import numpy as np
from nn.init import initialize

class Layer:
    """Base class for all neural network modules.
    You must implement forward and backward method to inherit this class.
    All the trainable parameters have to be stored in params and grads to be
    handled by the optimizer.
    """
    def __init__(self):
        self.params, self.grads = dict(), dict()

    def forward(self, *input):
        raise NotImplementedError
        
    def backward(self, *input):
        raise NotImplementedError


class Linear(Layer):
    """Linear (fully-connected) layer.

    Args:
        - in_dims (int): Input dimension of linear layer.
        - out_dims (int): Output dimension of linear layer.
        - init_mode (str): Weight initalize method. See `nn.init.py`.
          linear|normal|xavier|he are the possible options.
        - init_scale (float): Weight initalize scale for the normal init way.
          See `nn.init.py`.
        
    """
    def __init__(self, in_dims, out_dims, init_mode="linear", init_scale=1e-3):
        super().__init__()

        self.params["w"] = initialize((in_dims, out_dims), init_mode, init_scale)
        self.params["b"] = initialize(out_dims, "zero")
    
    def forward(self, x):
        """Calculate forward propagation.

        Returns:
            - out (numpy.ndarray): Output feature of this layer.
        """
        ######################################################################
        # TODO: Linear 레이어의 forward propagation 구현.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.x = x
        w = self.params["w"]
        b = self.params["b"]
        
        out = None
        N = x.shape[0]
        reshape_input = x.reshape(N, -1)
        
        out = np.dot(reshape_input, w) + b.T
        
        
        
        
        return out

    def backward(self, dout):
        """Calculate backward propagation.

        Args:
            - dout (numpy.ndarray): Derivative of output `out` of this layer.
        
        Returns:
            - dx (numpy.ndarray): Derivative of input `x` of this layer.
        """
        ######################################################################
        # TODO: Linear 레이어의 backward propagation 구현.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        x = self.x
        w = self.params["w"]
        b = self.params["b"]
        dx, dw, db = None, None, None
        
        N = x.shape[0]
        
        dx = np.dot(dout, w.T)
        
        dx = dx.reshape(x.shape)
        
        reshape_input = x.reshape(N, -1)
        dw = reshape_input.T.dot(dout)
        db = np.sum(dout, axis=0)
        
        self.grads["x"] = dx
        self.grads["w"] = dw
        self.grads["b"] = db
        return dx


class ReLU(Layer):
    def __init__(self):
        super().__init__()
                  

    def forward(self, x):
        out = None
        ######################################################################
        # TODO: ReLU 레이어의 forward propagation 구현.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.x = x
        out = np.maximum(0, x)
        self.out=out
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: ReLU 레이어의 backward propagation 구현.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        x = self.x
        mask = (x<=0)
        
        dout[mask] = 0
        dx = dout
        return dx


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = 0
        ######################################################################
        # TODO: Sigmoid 레이어의 forward propagation 구현.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.x=x
        out = 1/(1+np.exp(-x))
        self.out=out
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: Sigmoid 레이어의 backward propagation 구현.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        ###out = dout * (1 - dout)
        #f = 1/(1+np.exp(-self.x))
        dx = dout * (1 - self.out) * self.out
        self.grads["x"] = dx
        return dx


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ######################################################################
        # TODO: Tanh 레이어의 forward propagation 구현.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
   
        out = (np.exp(2*x)-1)/(np.exp(2*x)+1)
        self.out=out
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: Tanh 레이어의 backward propagation 구현.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        dx = dout * (1 - self.out) * (1 + self.out)
        self.grads["x"] = dx
        return dx


class SoftmaxCELoss(Layer):
    """Softmax and cross-entropy loss layer.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """Calculate both forward and backward propagation.
        
        Args:
            - x (numpy.ndarray): Pre-softmax (score) matrix (or vector).
            - y (numpy.ndarray): Label of the current data feature.

        Returns:
            - loss (float): Loss of current data.
            - dx (numpy.ndarray): Derivative of pre-softmax matrix (or vector).
        """
        ######################################################################
        # TODO: Softmax cross-entropy 레이어의 구현. 
        #        
        # NOTE: 이 메소드에서 forward/backward를 모두 수행하고, loss와 gradient (dx)를 
        # 리턴해야 함.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        #softmax 계산
        softmax_list = []
        for xi in x:
            exps = np.exp(xi - np.max(xi))
            s = exps / np.sum(exps)
            softmax_list.append(s)
        softmax = np.asarray(softmax_list)
#         print(softmax)
        #정답을 one-hot encoding
        #y=정답
        onehot_encoded = list()
        for value in y:
            letter = [0 for _ in range(softmax.shape[1])]
            letter[value] = 1
            onehot_encoded.append(letter)
        y = np.asarray(onehot_encoded)
        
        error = []
        dx = []
        for i in range(len(y)):
            correct = y[i, :]#t
            predict = softmax[i, :]#y
            total = 0
#             print('----------')
            for j in range(len(correct)):
                total = total + correct[j] * np.log(predict[j])
#                 print(total, correct[j], predict[j], np.log(predict[j]), correct[j] * np.log(predict[j]))
            error.append(-total)
            loss = np.mean(error)
#             print(error, loss)
            dx.append((predict - correct)/y.shape[0])

        dx = np.array(dx)

        return loss, dx
    
    
class Conv2d(Layer):
    """Convolution layer.

    Args:
        - in_dims (int): Input dimension of conv layer.
        - out_dims (int): Output dimension of conv layer.
        - ksize (int): Kernel size of conv layer.W
        - stride (int): Stride of conv layer.
        - pad (int): Number of padding of conv layer.
        - Other arguments are same as the Linear class.
    """
    def __init__(
        self, 
        in_dims, out_dims,
        ksize, stride, pad,
        init_mode="linear",
        init_scale=1e-3
    ):
        super().__init__()
        
        self.params["w"] = initialize(
            (out_dims, in_dims, ksize, ksize), 
            init_mode, init_scale)
        self.params["b"] = initialize(out_dims, "zero")
        
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        self.x =x
        N, C, H, W = x.shape
        w = self.params["w"]
        b = self.params["b"]
        F, C, HH, WW = w.shape
        P = self.pad
        S = self.stride
        
        H_R = 1 + (H + 2 * P - HH) / S
        W_R = 1 + (W + 2 * P - WW)/S
        
        H_R = int(H_R)
        W_R = int(W_R)
        out = np.zeros((N, F, H_R, W_R))
        
        x_pad = np.lib.pad(x, ((0,0), (0,0), (P, P), (P, P)), 'constant', constant_values=0)
        
        for n in range(N):
            for depth in range(F):
                for r in range(0, H, S):
                    for c in range(0, W, S):
                        r_S = int(r/S)
                        c_S = int(c/S)
                        out[n, depth, r_S, c_S] = np.sum(x_pad[n,:,r:r+HH,c:c+WW] * w[depth,:,:,:]) + b[depth]
                        
                   
        
        ######################################################################
        # TODO: Convolution 레이어의 forward propagation 구현.
        #
        # HINT: for-loop의 4-중첩으로 구현.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        x = self.x
        w = self.params["w"]
        b = self.params["b"]
        P = self.pad
        S = self.stride
        dx, dw, db = None, None, None
        N, F, H_R, W_R = dout.shape
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        
        x_pad = np.lib.pad(x, ((0, 0), (0,0), (P,P), (P,P)), 'constant', constant_values=0)
        dx = np.zeros(x_pad.shape)
        dw = np.zeros(w.shape)
        db = np.zeros(b.shape)
        
        for n in range(N):
            for depth in range(F):
                for r in range(0,H, S):
                    for c in range(0,W,S):
                        r_S = int(r/S)
                        c_S = int(c/S)
                        dx[n, :, r:r+HH, c:c+WW] += dout[n, depth, r_S, c_S]*w[depth,:,:,:]

        delete_rows = list(range(P)) + list(range(H+P, H+2*P, 1))
        delete_columns = list(range(P)) + list(range(W+P, W+2*P, 1))
        dx = np.delete(dx, delete_rows, axis = 2)
        dx = np.delete(dx, delete_columns, axis = 3)

        for n in range(N):
            for depth in range(F):
                for r in range(H_R):
                    for c in range(W_R):
                        dw[depth,:,:,:] += dout[n, depth, r, c] * x_pad[n,:, r*S:r*S+HH, c*S:c*S+WW]
        for depth in range(F):
            db[depth] = np.sum(dout[:,depth, :,:])
            
        ######################################################################
        # TODO: Convolution 레이어의 backward propagation 구현.
        #
        # HINT: for-loop의 4-중첩으로 구현.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        self.grads["w"] = dw
        self.grads["b"] = db
        return dx
    

class MaxPool2d(Layer):
    """Max pooling layer.

    Args:
        - ksize (int): Kernel size of maxpool layer.
        - stride (int): Stride of maxpool layer.
    """
    def __init__(self, ksize, stride):
        super().__init__()
        
        self.ksize = ksize
        self.stride = stride
        
    def forward(self, x):
           
        self.x = x
        N, C, H, W = x.shape
        out_h = int(1+(H - self.ksize) / self.stride)
        out_w = int(1+(W - self.ksize) / self.stride)
        

        N, C, H, W = x.shape
        pad = 0
        out_h = (H + 2*pad - self.ksize)//self.stride + 1
        out_w = (W + 2*pad - self.ksize)//self.stride + 1

        img = np.pad(x, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, self.ksize, self.ksize, out_h, out_w))

        for y in range(self.ksize):
            y_max = y + self.stride*out_h
            for x in range(self.ksize):
                x_max = x + self.stride*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:self.stride, x:x_max:self.stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        col = col.reshape(-1, self.ksize*self.ksize)

        out = np.max(col, axis = 1)

        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        
        
        ######################################################################
        # TODO: Max pooling 레이어의 forward propagation 구현.
        #
        # HINT: for-loop의 2-중첩으로 구현.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        x = self.x
        N, C, H, W = x.shape
        S = self.stride
        H_P = self.ksize
        W_P = self.ksize
        N, C, HH, WW = dout.shape
        
        dx = None
        dx = np.zeros(x.shape)
        
        for n in range(N):
            for depth in range(C):
                for r in range(HH):
                    for c in range(WW):
                        x_pool = x[n, depth, r*S:r*S+H_P, c*S:c*S+W_P]
                        mask = (x_pool == np.max(x_pool))
                        
                        dx[n, depth, r*S:r*S+H_P, c*S:c*S+W_P] = mask*dout[n, depth, r, c]
        ######################################################################
        # TODO: Max pooling 레이어의 backward propagation 구현.
        #
        # HINT: for-loop의 4-중첩으로 구현.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        return dx