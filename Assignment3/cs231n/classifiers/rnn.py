import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *

class CaptioningRNN():
    """
    CaptionningRNN 使用循环神经网络通过图像特征进行标注

    注意，我们不对 CaptioningRNN 使用任何的正则化
    """
    
    def __init__(self, 
                 word_to_idx,
                 input_dim = 512,
                 wordvec_dim = 128, 
                 hidden_dim = 128, 
                 cell_type = 'rnn', 
                 dtype = np.float32):
        """
        初始化一个 CaptioningRNN 实例

        输入：
        - word_to_idx：单词表字典，有 V 个入口，将每个字符串映射到 [0, V) 上一个唯一的整数
        - input—_dim：输入图像特征向量的维度 D
        - wordvec_dim：单词向量的维度 W
        - hidden_dim：RNN hidden state 的维度 H
        - cell_type：'rnn' or 'lstm'
        - dtype：数据类型
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)
    
        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}
        
        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)
        
        # Initialize word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100
        
        # Initialize CNN -> hidden state projection parameters
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)
        
        # Initialize output to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)
        
        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)
    
    def loss(self, features, captions):
        """
        计算 RNN 的训练损失。我么输入图像特征和对应的 ground-truth 标注
        然后使用 RNN（或 LSTM）计算损失和所有参数的梯度

        输入：
        - features：大小为 (N, D) 的图像特征
        - captions：ground_truth captions

        返回：
        - loss：损失值
        - grads：对应 self.params 的参数梯度字典
        """
            # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]
        
        # You'll need this 
        mask = (captions_out != self._null)

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        
        # Word embedding matrix
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
        
        loss, grads = 0.0, {}

        ############################################################################
        # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        ############################################################################
        #(1)
        affine_out, affine_cache = affine_forward(features ,W_proj, b_proj)
        #(2)
        word_embedding_out, word_embedding_cache = word_embedding_forward(captions_in, W_embed)
        #(3)
        if self.cell_type == 'rnn':
            rnn_or_lstm_out, rnn_cache = rnn_forward(word_embedding_out, affine_out, Wx, Wh, b)
        elif self.cell_type == 'lstm':
            rnn_or_lstm_out, lstm_cache = lstm_forward(word_embedding_out, affine_out, Wx, Wh, b)
        else:
            raise ValueError('Invalid cell_type "%s"' % self.cell_type)
        #(4)
        temporal_affine_out, temporal_affine_cache = temporal_affine_forward(rnn_or_lstm_out, W_vocab, b_vocab)
        #(5)
        loss, dtemporal_affine_out = temporal_softmax_loss(temporal_affine_out, captions_out, mask)
        #(4)
        drnn_or_lstm_out, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dtemporal_affine_out, temporal_affine_cache)
        #(3)
        if self.cell_type == 'rnn':
            dword_embedding_out, daffine_out, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(drnn_or_lstm_out, rnn_cache)
        elif self.cell_type == 'lstm':
            dword_embedding_out, daffine_out, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(drnn_or_lstm_out, lstm_cache)
        else:
            raise ValueError('Invalid cell_type "%s"' % self.cell_type)
        #(2)
        grads['W_embed'] = word_embedding_backward(dword_embedding_out, word_embedding_cache)
        #(1)
        dfeatures, grads['W_proj'], grads['b_proj'] = affine_backward(daffine_out, affine_cache)
        #pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        return loss, grads

    def sample(self, features, max_length = 30):
        """
        计算模型的测试时前向传播，同时对输入特征的标注进行采样

        在每一个时间步，我们嵌入当前的单词，然后将它和上一步的 hidden state 传给 RNN，
        使用 hidden state 为所有的单词表单词评分，再之后选择得分最高的词作为下一步
        传入的词。初始的 hidden state 通过对输入的图像特征进行全连接映射得到，初始的单词用 <Start> 标记
        
        对于 LSTM，我们还需要记录 cell state，初始时的 cell state 应该为 0

        输入：
        - features：大小为 (N, D) 的输入图像特征
        - max_length：产生的标注的最大长度 T

        返回：
        - captions：大小为 (N, max_length) 的数组，给出采样的标注，其中每个元素是一个在范围 [0, V) 上的
        整数，标注的第一个元素应该是第一个采样的单词，而不是 <START> 标记
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
        
        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     to the appropriate slot in the captions variable                    #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        ###########################################################################
        N, D = features.shape
        affine_out, affine_cache = affine_forward(features ,W_proj, b_proj)
        
        prev_word_idx = [self._start]*N
        prev_h = affine_out
        prev_c = np.zeros(prev_h.shape)
        captions[:,0] = self._start
        for i in range(1,max_length):
            prev_word_embed  = W_embed[prev_word_idx]
            if self.cell_type == 'rnn':
                next_h, rnn_step_cache = rnn_step_forward(prev_word_embed, prev_h, Wx, Wh, b)
            elif self.cell_type == 'lstm':
                next_h, next_c,lstm_step_cache = lstm_step_forward(prev_word_embed, prev_h, prev_c, Wx, Wh, b)
                prev_c = next_c
            else:
                raise ValueError('Invalid cell_type "%s"' % self.cell_type)
            vocab_affine_out, vocab_affine_out_cache = affine_forward(next_h, W_vocab, b_vocab)
            captions[:,i] = list(np.argmax(vocab_affine_out, axis = 1))
            prev_word_idx = captions[:,i]
            prev_h = next_h
        #pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions
