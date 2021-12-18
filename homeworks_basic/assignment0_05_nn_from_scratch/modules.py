#!/usr/bin/env python
# coding: utf-8

# Credits: this notebook belongs to [Practical DL](https://docs.google.com/forms/d/e/1FAIpQLScvrVtuwrHSlxWqHnLt1V-_7h2eON_mlRR6MUb3xEe5x9LuoA/viewform?usp=sf_link) course by Yandex School of Data Analysis.

# In[167]:


import numpy as np


# **Module** is an abstract class which defines fundamental methods necessary for a training a neural network. You do not need to change anything here, just read the comments.

# In[168]:


class Module(object):
    """
    Basically, you can think of a module as of a something (black box) 
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`: 
        
        output = module.forward(input)
    
    The module should be able to perform a backward pass: to differentiate the `forward` function. 
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule. 
    
        gradInput = module.backward(input, gradOutput)
    """
    def __init__ (self):
        self.output = None
        self.gradInput = None
        self.training = True
    
    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        return self.updateOutput(input)

    def backward(self,input, gradOutput):
        """
        Performs a backpropagation step through the module, with respect to the given input.
        
        This includes 
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput
    

    def updateOutput(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the `output` field.
        
        Make sure to both store the data in `output` field and return it. 
        """
        
        # The easiest case:
            
        # self.output = input 
        # return self.output
        
        pass

    def updateGradInput(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own input. 
        This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.
        
        The shape of `gradInput` is always the same as the shape of `input`.
        
        Make sure to both store the gradients in `gradInput` field and return it.
        """
        
        # The easiest case:
        
        # self.gradInput = gradOutput 
        # return self.gradInput
        
        pass   
    
    def accGradParameters(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass
    
    def zeroGradParameters(self): 
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass
        
    def getParameters(self):
        """
        Returns a list with its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
        
    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
    
    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True
    
    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False
    
    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Module"


# # Sequential container

# **Define** a forward and backward pass procedures.

# In[5]:


class Sequential(Module):
    """
         This class implements a container, which processes `input` data sequentially. 
         
         `input` is processed by each module (layer) in self.modules consecutively.
         The resulting array is called `output`. 
    """
    
    def __init__ (self):
        super(Sequential, self).__init__()
        self.modules = []
   
    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, input):
        """
        Basic workflow of FORWARD PASS:
        
            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})   
            
            
        Just write a little loop. 
        """

        # Your code goes here. ################################################
        self.output = input
        for module in self.modules:
            self.output = module.forward(self.output)
        return self.output

    def backward(self, input, gradOutput):
        """
        Workflow of BACKWARD PASS:
            
            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)   
            gradInput = module[0].backward(input, g_1)   
             
             
        !!!
                
        To ech module you need to provide the input, module saw while forward pass, 
        it is used while computing gradients. 
        Make sure that the input for `i-th` layer the output of `module[i]` (just the same input as in forward pass) 
        and NOT `input` to this Sequential module. 
        
        !!!
        
        """
        # Your code goes here. ################################################
        self.gradInput = gradOutput
        for i in range(len(self.modules)-1, 0, -1):
            self.gradInput = self.modules[i].backward(self.modules[i-1].output, self.gradInput)
        self.gradInput = self.modules[0].backward(input, self.gradInput)
        return self.gradInput
      

    def zeroGradParameters(self): 
        for module in self.modules:
            module.zeroGradParameters()
    
    def getParameters(self):
        """
        Should gather all parameters in a list.
        """
        return [x.getParameters() for x in self.modules]
    
    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.getGradParameters() for x in self.modules]
    
    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string
    
    def __getitem__(self,x):
        return self.modules.__getitem__(x)
    
    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.modules:
            module.train()
    
    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.modules:
            module.evaluate()


# # Layers

# ## 1. Linear transform layer
# Also known as dense layer, fully-connected layer, FC-layer, InnerProductLayer (in caffe), affine transform
# - input:   **`batch_size x n_feats1`**
# - output: **`batch_size x n_feats2`**

# In[48]:


class Linear(Module):
    """
    A module which applies a linear transformation 
    A common name is fully-connected layer, InnerProductLayer in caffe. 
    
    The module should work with 2D input of shape (n_samples, n_feature).
    """
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
       
        # This is a nice initialization
        stdv = 1./np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size = n_out)
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def updateOutput(self, input):
        # Your code goes here. ################################################
        self.output = np.add(input.dot(self.W.T), self.b)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradInput = np.dot(gradOutput, self.W)
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradW += np.sum(input[:, None, :] * gradOutput[:, :, None], axis=0)
        self.gradb += gradOutput.sum(axis=0)
        pass
    
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' %(s[1],s[0])
        return q


# ## 2. SoftMax
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**
# 
# $\text{softmax}(x)_i = \frac{\exp x_i} {\sum_j \exp x_j}$
# 
# Recall that $\text{softmax}(x) == \text{softmax}(x - \text{const})$. It makes possible to avoid computing exp() from large argument.

# In[46]:


class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        
        # Your code goes here. ################################################
        self.output = np.exp(self.output)
        self.output = self.output / np.sum(self.output, axis=1, keepdims=True)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        deriv = np.zeros((gradOutput.shape[0], gradOutput.shape[1], gradOutput.shape[1]))
        softmax = self.output
        for i in range(gradOutput.shape[0]):
            deriv[i] = np.diag(softmax[i])
            deriv[i] -= softmax[i].reshape(-1, 1).dot(softmax[i].reshape(-1, 1).T)
        self.gradInput = np.matmul(gradOutput[:, None, :], (deriv))[:, 0, :]
        return self.gradInput
    
    def __repr__(self):
        return "SoftMax"


# ## 3. LogSoftMax
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**
# 
# $\text{logsoftmax}(x)_i = \log\text{softmax}(x)_i = x_i - \log {\sum_j \exp x_j}$
# 
# The main goal of this layer is to be used in computation of log-likelihood loss.

# In[47]:


class LogSoftMax(Module):
    def __init__(self):
         super(LogSoftMax, self).__init__()
    
    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        
        # Your code goes here. ################################################
        sums = np.matrix(np.log(np.sum(np.exp(self.output), axis=1)))
        self.output = np.array(self.output - np.transpose(sums))
        
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        output = np.subtract(input, input.max(axis=1, keepdims=True))
        
        exps = np.exp(output)
        sums = np.matrix(np.sum(exps, axis=1))
        
        output = exps / np.transpose(sums)
                
        batch_size = output.shape[0]
        output_size = output.shape[1]
        self.gradInput = []
        
        for i in range(batch_size):
            tmp = np.eye(output_size)
            for j in range(output_size):
                for k in range(output_size):
                    tmp[j, k] = tmp[j, k] - output[i, k]
            self.gradInput.append(np.dot(gradOutput[i], tmp))
            
        self.gradInput = np.array(self.gradInput)
        return self.gradInput

    
    def __repr__(self):
        return "LogSoftMax"


# ## 4. Batch normalization
# One of the most significant recent ideas that impacted NNs a lot is [**Batch normalization**](http://arxiv.org/abs/1502.03167). The idea is simple, yet effective: the features should be whitened ($mean = 0$, $std = 1$) all the way through NN. This improves the convergence for deep models letting it train them for days but not weeks. **You are** to implement the first part of the layer: features normalization. The second part (`ChannelwiseScaling` layer) is implemented below.
# 
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**
# 
# The layer should work as follows. While training (`self.training == True`) it transforms input as $$y = \frac{x - \mu}  {\sqrt{\sigma + \epsilon}}$$
# where $\mu$ and $\sigma$ - mean and variance of feature values in **batch** and $\epsilon$ is just a small number for numericall stability. Also during training, layer should maintain exponential moving average values for mean and variance: 
# ```
#     self.moving_mean = self.moving_mean * alpha + batch_mean * (1 - alpha)
#     self.moving_variance = self.moving_variance * alpha + batch_variance * (1 - alpha)
# ```
# During testing (`self.training == False`) the layer normalizes input using moving_mean and moving_variance. 
# 
# Note that decomposition of batch normalization on normalization itself and channelwise scaling here is just a common **implementation** choice. In general "batch normalization" always assumes normalization + scaling.

# In[49]:


class BatchNormalization(Module):
    EPS = 1e-3
    def __init__(self, alpha = 0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = 0
        self.moving_variance = 0
        
    def updateOutput(self, input):
        ## Your code goes here. ################################################??????????????????????
        # use self.EPS please
        batch_mean = self.moving_mean
        batch_variance = self.moving_variance
        
        if self.training == True:
            batch_mean = np.mean(input, axis=0)
            batch_variance = np.var(input, axis=0)
            self.moving_mean = self.moving_mean * self.alpha + batch_mean * (1 - self.alpha)
            self.moving_variance = self.moving_variance * self.alpha + batch_variance * (1 - self.alpha)
                   
        self.output = (input - batch_mean) / np.sqrt(batch_variance + self.EPS)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        ## Your code goes here. ################################################
        h = input
        dy = gradOutput
        eps = self.EPS
        N = input.shape[0]
        
        mean = 1 / N * np.sum(h, axis=0)
        var = 1 / N * np.sum((h - mean)**2, axis=0)
        self.gradInput = (1 / N) * (var + eps)**(-1. / 2) * (N * dy - np.sum(dy, axis=0)
                - (h - mean) * (var + eps)**(-1.0) * np.sum(dy * (h - mean), axis=0))
        return self.gradInput
    
    def __repr__(self):
        return "BatchNormalization"


# In[14]:


class ChannelwiseScaling(Module):
    """
       Implements linear transform of input y = \gamma * x + \beta
       where \gamma, \beta - learnable vectors of length x.shape[-1]
    """
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)
        
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output
        
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        self.gradBeta = np.sum(gradOutput, axis=0)
        self.gradGamma = np.sum(gradOutput*input, axis=0)
    
    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)
        
    def getParameters(self):
        return [self.gamma, self.beta]
    
    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]
    
    def __repr__(self):
        return "ChannelwiseScaling"


# Practical notes. If BatchNormalization is placed after a linear transformation layer (including dense layer, convolutions, channelwise scaling) that implements function like `y = weight * x + bias`, than bias adding become useless and could be omitted since its effect will be discarded while batch mean subtraction. If BatchNormalization (followed by `ChannelwiseScaling`) is placed before a layer that propagates scale (including ReLU, LeakyReLU) followed by any linear transformation layer than parameter `gamma` in `ChannelwiseScaling` could be freezed since it could be absorbed into the linear transformation layer.

# ## 5. Dropout
# Implement [**dropout**](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf). The idea and implementation is really simple: just multimply the input by $Bernoulli(p)$ mask. Here $p$ is probability of an element to be zeroed.
# 
# This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons.
# 
# While training (`self.training == True`) it should sample a mask on each iteration (for every batch), zero out elements and multiply elements by $1 / (1 - p)$. The latter is needed for keeping mean values of features close to mean values which will be in test mode. When testing this module should implement identity transform i.e. `self.output = input`.
# 
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**

# In[41]:


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        
        self.p = p
        self.mask = None
        
    def updateOutput(self, input):
        # Your code goes here. ################################################
        if not self.training:
            self.output = input
            return self.output
        from scipy.stats import bernoulli
        self.mask = bernoulli.rvs((1.0 - self.p) * np.ones(input.shape))
        coef = np.multiply(self.mask, 1 / (1 - self.p))
        self.output = np.multiply(input, coef)
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradInput =  np.multiply(gradOutput, np.multiply(self.mask, 1 / (1 - self.p)))
        return self.gradInput
        
    def __repr__(self):
        return "Dropout"


# # Activation functions

# Here's the complete example for the **Rectified Linear Unit** non-linearity (aka **ReLU**): 

# In[8]:


class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput , input > 0)
        return self.gradInput
    
    def __repr__(self):
        return "ReLU"


# ## 6. Leaky ReLU
# Implement [**Leaky Rectified Linear Unit**](http://en.wikipedia.org/wiki%2FRectifier_%28neural_networks%29%23Leaky_ReLUs). Expriment with slope. 

# In[9]:


class LeakyReLU(Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()
            
        self.slope = slope
        
    def updateOutput(self, input):
        # Your code goes here. ################################################
        f = np.vectorize(lambda x: x if x > 0 else self.slope * x, otypes=[np.float]) 
        self.output = f(input)
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        f = np.vectorize(lambda x: 1 if x > 0 else self.slope, otypes=[np.float]) 
        self.gradInput = np.multiply(gradOutput, f(input))
        return self.gradInput
    
    def __repr__(self):
        return "LeakyReLU"


# ## 7. ELU
# Implement [**Exponential Linear Units**](http://arxiv.org/abs/1511.07289) activations.

# In[10]:


class ELU(Module):
    def __init__(self, alpha = 1.0):
        super(ELU, self).__init__()
        
        self.alpha = alpha
        
    def updateOutput(self, input):
        # Your code goes here. ################################################
        f = np.vectorize(lambda x: x if x > 0 else self.alpha * (np.exp(x) - 1), otypes=[np.float]) 
        self.output = f(input)
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        f = np.vectorize(lambda x: 1 if x > 0 else self.alpha * np.exp(x), otypes=[np.float]) 
        self.gradInput = np.multiply(gradOutput, f(input))
        return self.gradInput
    
    def __repr__(self):
        return "ELU"


# ## 8. SoftPlus
# Implement [**SoftPlus**](https://en.wikipedia.org/wiki%2FRectifier_%28neural_networks%29) activations. Look, how they look a lot like ReLU.

# In[11]:


class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()
    
    def updateOutput(self, input):
        # Your code goes here. ################################################
        self.output = np.log((1 + np.exp(input)))
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradInput = np.multiply(gradOutput, 1 / (1 + np.exp(-input)))
        return self.gradInput
    
    def __repr__(self):
        return "SoftPlus"


# # Criterions

# Criterions are used to score the models answers. 

# In[7]:


class Criterion(object):
    def __init__ (self):
        self.output = None
        self.gradInput = None
        
    def forward(self, input, target):
        """
            Given an input and a target, compute the loss function 
            associated to the criterion and return the result.
            
            For consistency this function should not be overrided,
            all the code goes in `updateOutput`.
        """
        return self.updateOutput(input, target)

    def backward(self, input, target):
        """
            Given an input and a target, compute the gradients of the loss function
            associated to the criterion and return the result. 

            For consistency this function should not be overrided,
            all the code goes in `updateGradInput`.
        """
        return self.updateGradInput(input, target)
    
    def updateOutput(self, input, target):
        """
        Function to override.
        """
        return self.output

    def updateGradInput(self, input, target):
        """
        Function to override.
        """
        return self.gradInput   

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Criterion"


# The **MSECriterion**, which is basic L2 norm usually used for regression, is implemented here for you.
# - input:   **`batch_size x n_feats`**
# - target: **`batch_size x n_feats`**
# - output: **scalar**

# In[8]:


class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()
        
    def updateOutput(self, input, target):   
        self.output = np.sum(np.power(input - target,2)) / input.shape[0]
        return self.output 
 
    def updateGradInput(self, input, target):
        self.gradInput  = (input - target) * 2 / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"


# ## 9. Negative LogLikelihood criterion (numerically unstable)
# You task is to implement the **ClassNLLCriterion**. It should implement [multiclass log loss](http://scikit-learn.org/stable/modules/model_evaluation.html#log-loss). Nevertheless there is a sum over `y` (target) in that formula, 
# remember that targets are one-hot encoded. This fact simplifies the computations a lot. Note, that criterions are the only places, where you divide by batch size. Also there is a small hack with adding small number to probabilities to avoid computing log(0).
# - input:   **`batch_size x n_feats`** - probabilities
# - target: **`batch_size x n_feats`** - one-hot representation of ground truth
# - output: **scalar**
# 
# 

# In[10]:


class ClassNLLCriterionUnstable(Criterion):
    EPS = 1e-15
    def __init__(self):
        a = super(ClassNLLCriterionUnstable, self)
        super(ClassNLLCriterionUnstable, self).__init__()
        
    def updateOutput(self, input, target): 
        
        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        
        # Your code goes here. ################################################
        self.output = (-1 / target.shape[0]) * np.sum(target * np.log(input_clamp))
        return self.output

    def updateGradInput(self, input, target):
        
        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
                
        # Your code goes here. ################################################
        local_dev = -1 / (input_clamp * target.shape[0])
        self.gradInput =  local_dev * target
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterionUnstable"


# ## 10. Negative LogLikelihood criterion (numerically stable)
# - input:   **`batch_size x n_feats`** - log probabilities
# - target: **`batch_size x n_feats`** - one-hot representation of ground truth
# - output: **scalar**
# 
# Task is similar to the previous one, but now the criterion input is the output of log-softmax layer. This decomposition allows us to avoid problems with computation of forward and backward of log().

# In[11]:


class ClassNLLCriterion(Criterion):
    def __init__(self):
        a = super(ClassNLLCriterion, self)
        super(ClassNLLCriterion, self).__init__()
        
    def updateOutput(self, input, target): 
        # Your code goes here. ################################################
        self.output = (-1 / target.shape[0]) * np.sum(target * input)
        return self.output

    def updateGradInput(self, input, target):
        # Your code goes here. ################################################
        local_dev = -1 / target.shape[0]
        self.gradInput = local_dev * target
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterion"


# # Optimizers

# ### SGD optimizer with momentum
# - `variables` - list of lists of variables (one list per layer)
# - `gradients` - list of lists of current gradients (same structure as for `variables`, one array for each var)
# - `config` - dict with optimization parameters (`learning_rate` and `momentum`)
# - `state` - dict with optimizator state (used to save accumulated gradients)

# In[ ]:


def sgd_momentum(variables, gradients, config, state):  
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault('accumulated_grads', {})
    
    var_index = 0 
    for current_layer_vars, current_layer_grads in zip(variables, gradients): 
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            
            old_grad = state['accumulated_grads'].setdefault(var_index, np.zeros_like(current_grad))
            
            np.add(config['momentum'] * old_grad, config['learning_rate'] * current_grad, out=old_grad)
            
            current_var -= old_grad
            var_index += 1     


# ## 11. [Adam](https://arxiv.org/pdf/1412.6980.pdf) optimizer
# - `variables` - list of lists of variables (one list per layer)
# - `gradients` - list of lists of current gradients (same structure as for `variables`, one array for each var)
# - `config` - dict with optimization parameters (`learning_rate`, `beta1`, `beta2`, `epsilon`)
# - `state` - dict with optimizator state (used to save 1st and 2nd moment for vars)
# 
# Formulas for optimizer:
# 
# Current step learning rate: $$\text{lr}_t = \text{learning_rate} * \frac{\sqrt{1-\beta_2^t}} {1-\beta_1^t}$$
# First moment of var: $$\mu_t = \beta_1 * \mu_{t-1} + (1 - \beta_1)*g$$ 
# Second moment of var: $$v_t = \beta_2 * v_{t-1} + (1 - \beta_2)*g*g$$
# New values of var: $$\text{variable} = \text{variable} - \text{lr}_t * \frac{m_t}{\sqrt{v_t} + \epsilon}$$

# In[50]:


def adam_optimizer(variables, gradients, config, state):  
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault('m', {})  # first moment vars
    state.setdefault('v', {})  # second moment vars
    state.setdefault('t', 0)   # timestamp
    state['t'] += 1
    for k in ['learning_rate', 'beta1', 'beta2', 'epsilon']:
        assert k in config, config.keys()
    
    var_index = 0 
    lr_t = config['learning_rate'] * np.sqrt(1 - config['beta2']**state['t']) / (1 - config['beta1']**state['t'])
    for current_layer_vars, current_layer_grads in zip(variables, gradients): 
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            var_first_moment = state['m'].setdefault(var_index, np.zeros_like(current_grad))
            var_second_moment = state['v'].setdefault(var_index, np.zeros_like(current_grad))
            
            # <YOUR CODE> #######################################
            # update `current_var_first_moment`, `var_second_moment` and `current_var` values
            #np.add(... , out=var_first_moment)
            #np.add(... , out=var_second_moment)
            #current_var -= ...
            np.add(config['beta1'] * var_first_moment, (1 - config['beta1']) * current_grad, out=var_first_moment)
            np.add(config['beta2'] * var_second_moment, (1 - config['beta2']) * current_grad ** 2, out=var_second_moment)
            current_var -= lr_t * var_first_moment / (np.sqrt(var_second_moment) + config['epsilon'])
            
            # small checks that you've updated the state; use np.add for rewriting np.arrays values
            assert var_first_moment is state['m'].get(var_index)
            assert var_second_moment is state['v'].get(var_index)
            var_index += 1
