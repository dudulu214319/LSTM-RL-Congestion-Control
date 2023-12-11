import tensorflow as tf
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data

# 设置随机数种子
tf.random.set_seed(1)
np.random.seed(1)

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, InputSpec

class Self_Attention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape): # input_shape[2]是attention前一层的输出维度
        print("input_shape:", input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Self_Attention, self).build(input_shape)

    def call(self, x):
        print('x:', x)
        print('kernel:', self.kernel)
        print('kernel[0]:', self.kernel[0])
        print('kernel[1]:', self.kernel[1])
        print('kernel[2]:', self.kernel[2])
        WQ = K.dot(x, self.kernel[0])
        print('WQ:', WQ)
        WK = K.dot(x, self.kernel[1])
        print('WK', WK)
        WV = K.dot(x, self.kernel[2])
        print('WV', WV)
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        print('QK', QK)
        QK = QK / (64 ** 0.5)
        QK = K.softmax(QK)
        V = K.batch_dot(QK, WV)
        print('V:', V)
        return V

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_dim


class Final_Attention(nn.Module):
    def __init__(self):
        super(Final_Attention, self).__init__()
        self.out = nn.Linear(100, 1)

    def attention_net(self, lstm_output, final_state):
        # lstm_output : [batch_size, n_step, n_hidden * num_directions(=1)], F matrix
        # final_state : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]

        batch_size = len(lstm_output)
        hidden = final_state.view(batch_size,-1,1)
        # hidden = torch.cat((final_state[0], final_state[1]), dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=1), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        # attn_weights : [batch_size,n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)

        # context: [batch_size, n_hidden * num_directions(=1)]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return context, soft_attn_weights

    def forward(self, output, final_hidden_state):
        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        # output : [seq_len, batch_size, n_hidden * num_directions(=1)]

        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output),attention # attn_output : [batch_size, num_classes], attention : [batch_size, n_step]