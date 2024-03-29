import numpy as np
import nn.layers as layers

class TwoLayerNet:
    """A neural network that has two fully-connected (linear) layers.
    This model can be illustrated as:
    `input -> linear@hidden_dim -> relu -> linear@num_classes -> softmax`
    Here, linear@X represents linear layer that has `X` output dimension.

    Args:
        - input_dim (int): Input dimension.
        - hidden_dim (int): Hidden dimension. 
          It should be output dimension of first linear layer.
        - num_classes (int): Number of classes, and it should be output 
          dimension of second (last) linear layer.
        - init_mode (str): Weight initalize method. See `nn.init.py`.
          linear|normal|xavier|he are the possible options.
        - init_scale (float): Weight initalize scale for the normal init way.
          See `nn.init.py`.
    """
    def __init__(
        self, 
        input_dim=3*32*32, 
        hidden_dim=100, 
        num_classes=10,
        init_mode="linear",
        init_scale=1e-3
):
        self.modules = dict()
        self.modules["linear1"] = layers.Linear(input_dim,hidden_dim,init_mode=init_mode)
        self.modules["relu1"] = layers.ReLU()
        self.modules["linear2"] = layers.Linear(hidden_dim,num_classes,init_mode=init_mode)
        self.modules["softmax"] = layers.SoftmaxCELoss()
        
        ######################################################################
        # TODO: 2-레이어 네트워크에 필요한 모듈들을 초기화. 필요한 모듈은
        # input -> linear -> relu -> linear -> softmax 와 같으며, 첫번째
        # linear 레이어는 hidden_dim을 출력 dimension으로, 두번째 레이어는
        # num_classes를 출력 dimension으로 구현해야 함.
        #
        # NOTE: 모든 레이어는 self.modules에 적절한 이름을 (e.g. "fc1")
        # key 값으로 사용하여 저장되어야 함.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
    
    def loss(self, X, y=None):
        """Compute loss and gradient for a minibatch of data.

        Args:
            - X: Array of input data of shape (N, C), where N is batch size 
              and C is input_dim.
            - y: Array of labels of shape (N,). y[i] gives the label for X[i].

        Return:
            - loss: Loss for a current minibatch of data.
        """
        scores = None
        ######################################################################
        # TODO: 현재 모델의 forward propagation을 구현. Softmax 레이어의 이전
        # 값인 scores를 계산하고, 이를 scores 변수에 저장해야 함.
        ######################################################################
        for key, layer in self.modules.items():
            if key == 'softmax':
                pass
            else:
                X = layer.forward(X)
        scores = X
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        if y is None:
            return scores
        ######################################################################
        # TODO: Backward propagation을 구현. Softmax cross entropy 레이어의
        # 출력 결과인 loss를 loss 변수에 저장하며, 두번째 리턴값인 출력의
        # derivative를 사용하여 backward 연산을 역순으로 진행해야 함
        ######################################################################
        loss, dx = layer.forward(scores, y)
        
        layers = list(self.modules.values())[:-1] # 마지막 레이어 제외
        layers.reverse()
        for layer in layers:
            dx = layer.backward(dx)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return loss


class FCNet:
    """A neural network that has arbitrary number of layers.
    This model can be illustrated as:
    `input -> linear -> relu -> linear -> ... -> linear -> softmax`

    Args:
        - hidden_dims (list): Hidden dimensions of layers.
          Each element are the output dimension of i-th fc layer.
          So that, total #layers = len(hidden_dims) + 1
        - Other arguments are same as the TwoLayerNet.
    """
    def __init__(
        self, 
        input_dim=3*32*32, 
        hidden_dims=[100, 100, 100], 
        num_classes=10,
        init_mode="linear",
        init_scale=1e-3
):
        self.modules = dict()
        self.num_layers = 1 + len(hidden_dims)
        
        # input layer
        self.modules["linear1"] = layers.Linear(input_dim,hidden_dims[0],init_mode=init_mode)
        self.modules["relu1"] = layers.ReLU()
        # hidden layers
        if self.num_layers > 2: # 히든 레이어가 하나 이상 있는 경우
            for i in range(self.num_layers - 2): # -2는 인풋 레이어와 아웃풋 레이어 개수 뺀 것
                self.modules["linear" + str(i+2)] = layers.Linear(hidden_dims[i],hidden_dims[i+1],init_mode=init_mode)
                self.modules["relu" + str(i+2)] = layers.ReLU()
        else:
            raise
        # output layer
        self.modules["linear3"] = layers.Linear(hidden_dims[-1],num_classes,init_mode=init_mode)
        self.modules["softmax"] = layers.SoftmaxCELoss()
        
        ######################################################################
        # TODO: 임의의 레이어를 가지는 FCNet에 필요한 모듈들을 초기화.
        #
        # NOTE: 모든 레이어는 self.modules에 적절한 이름을 (e.g. "fc1") key로 저장.
        #
        # HINT: 임의의 레이어를 처리하기 위해 for loop를 사용해야 함.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
    
    def loss(self, X, y=None):
        """Compute loss and gradient for a minibatch of data.
        Args and Returns are same as the TwoLayerNet.
        """
        scores = None
        ######################################################################
        # TODO: 현재 모델의 forward propagation을 구현. Softmax 레이어의 이전
        # 값인 scores를 계산하고, 이를 scores 변수에 저장해야 함.
        #
        # HINT: 임의의 레이어를 처리하기 위해 for loop를 사용해야 함.
        ######################################################################
        for key, layer in self.modules.items():
            if key == 'softmax':
                pass
            else:
                X = layer.forward(X)
        scores = X
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        if y is None:
            return scores
        ######################################################################
        # TODO: Backward propagation을 구현. Softmax cross entropy 레이어의
        # 출력 결과인 loss를 loss 변수에 저장하며, 두번째 리턴값인 출력의
        # derivative를 사용하여 backward 연산을 역순으로 진행해야 함.
        #
        # HINT: 임의의 레이어를 처리하기 위해 for loop를 사용해야 함.
        ######################################################################
        loss, dx = layer.forward(scores, y)
        
        layers = list(self.modules.values())[:-1] # 마지막 레이어 제외
        layers.reverse()
        for layer in layers:
            dx = layer.backward(dx)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return loss
        
        
class ThreeLayerConvNet:
    """A neural network that has one conv and two linear layers.
    This model can be illustrated as:
    `input -> conv@num_filters -> relu -> pool -> linear@hidden_dim ->
    relu -> linear@num_classes -> softmax`.
    Here, linear@X represents linear layer that has `X` output dimension and
    conv@X shows conv layer with `X` number of filters.

    Unlike FCNet, the network operates on minibatches of data have 
    shape (N, C, H, W) consisting of N images (batch size), each with 
    height H and width W and with C input channels.

    Args:
        - input_dim (list or tuple): Input dimension of single input 
          **image**. Normally, it could be (C, H, W) dimension.
        - num_filters (int): Number of filters (channels) of conv layer.
        - ksize (int): Kernel size of conv layer.
        - stride (int): Stride of conv layer.
        - pad (int): Number of padding of conv layer.
        - Other arguments are same as the TwoLayerNet.
    """
    def __init__(
        self, 
        input_dim=[3,32,32], 
        num_filters=32,
        ksize=3, stride=1, pad=1,
        hidden_dim=100, 
        num_classes=10,
        init_mode="linear",
        init_scale=1e-3
):
        self.modules = dict()
        
        # input layer
        self.modules["conv1"] = layers.Conv2d(in_dims=input_dim[0], out_dims=num_filters, ksize=ksize, stride=stride, pad=pad, init_mode=init_mode, init_scale=1e-3)
        self.modules["relu1"] = layers.ReLU()
        self.modules["pool1"] = layers.MaxPool2d(ksize = 2, stride = 2)
        
        self.modules["linear2"] = layers.Linear(num_filters*int(input_dim[1]/2)*int(input_dim[2]/2), hidden_dim, init_mode=init_mode)
        self.modules["relu2"] = layers.ReLU()
        
        self.modules["linear3"] = layers.Linear(hidden_dim, num_classes, init_mode=init_mode)
        self.modules["softmax"] = layers.SoftmaxCELoss()
        ######################################################################
        # TODO: 3-레이어 conv 네트워크에 필요한 모듈들을 초기화. 필요한 모듈은
        # input -> conv -> relu -> pool -> linear -> relu -> linear -> softmax
        # 와 같음. 첫번째 conv 레이어는 num_filters, ksize, stride 와 pad를 인자로 받음.
        # pool 레이어는 2x2 max pool을 사용하며, 첫번째 linear 레이어는 hidden_dim을 
        # 출력 dimension으로, 두번째 레이어는 num_classes를 출력 dimension으로 구현해야 함.
        #
        # NOTE: 모든 레이어는 self.modules에 적절한 이름을 (e.g. "conv1")
        # key값으로 사용하여 저장되어야 함.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
    
    def loss(self, X, y=None):
        """Compute loss and gradient for a minibatch of data.

        Args:
            - X: Array of input **image** data of shape (N, C, H, W), where
              N is batch size, C is number of channels, height and for width.
            - y: Array of labels of shape (N,). y[i] gives the label for X[i].

        Return:
            - loss: Loss for a current minibatch of data.
        """
        scores = None
        ######################################################################
        # TODO: 현재 모델의 forward propagation을 구현. Softmax 레이어의 이전값인 
        # scores를 계산하고, 이를 scores 변수에 저장해야 함.
        #
        # HINT: Linear forward 의 입력을 위해 feature를  flatten 시켜야 함.
        # e.g. (N, C, H, W) -> (N, C*H*W)
        ######################################################################
        for key, layer in self.modules.items():
            if key == 'softmax':
                pass
            else:
                if key == 'linear2':
                    shape_tmp = X.shape
                    X.reshape(shape_tmp[0], shape_tmp[1]*shape_tmp[2]*shape_tmp[3])
                X = layer.forward(X)
        scores = X

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        if y is None:
            return scores
        ######################################################################
        # TODO: Backward propagation을 구현. Softmax cross entropy 레이어의
        # 출력 결과인 loss를 loss 변수에 저장하며, 두번째 리턴값인 출력의
        # derivative를 사용하여 backward 연산을 역순으로 진행해야 함.
        #
        # HINT: Conv backward의 입력을 reshape 해야함 (forward시 미리 저장)
        # e.g. (N, C*H*W) -> (N, C, H, W)
        ######################################################################
        loss, dx = layer.forward(scores, y)
        
        layers = list(self.modules.values())[:-1] # 마지막 레이어 제외
        layers.reverse()
        for layer in layers:
            if key == 'conv1':
                X.reshape(shape_tmp[0], shape_tmp[1], shape_tmp[2], shape_tmp[3])
            dx = layer.backward(dx)

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return loss
