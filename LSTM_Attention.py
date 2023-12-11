# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import SGD,Adagrad
# from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation,LSTM
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from Attention_model import Self_Attention, Final_Attention
from sklearn.preprocessing import MinMaxScaler

# 设置随机数种子
# tf.random.set_seed(1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
np.random.seed(1)
#定义时间步
time_steps=5
batch_size = 16
input_window = 100
output_window = 1
scaler1 = MinMaxScaler() #用于归一化x
scaler2 = MinMaxScaler() #用于归一化y
scaler = MinMaxScaler(feature_range=(0, 1))

#定义模型
# def build_model(x_train,y_train,x_test,y_test):
#     model = Sequential()
#     model.add(LSTM(100, input_shape=(time_steps, 1), activation='relu', return_sequences=True))
#     model.add(LSTM(100, activation='relu', return_sequences=True))
#     model.add(Self_Attention(3)) #自注意力层，将信息attention到3维，里面有函数自行捕获input_shape
#     model.add(Flatten())  # Flatte层
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer=Adagrad(learning_rate=0.02))
#     history = model.fit(x_train, y_train, epochs=100, batch_size=16, validation_data=(x_test, y_test), verbose=0,
#                         shuffle=False)
#     return model, history

class LSTM_Attention(nn.Module):
    def __init__(self):
        super(LSTM_Attention, self).__init__()
        self.lstm = nn.LSTM(1, 100, bidirectional=False)
        self.attention_network = Final_Attention().to(device)

    def forward(self,X): # input : [batch_size, seq_len, embedding_dim]
        input = X.transpose(0, 1) # input : [seq_len, batch_size, embedding_dim]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        # output : [seq_len, batch_size, n_hidden * num_directions(=2)]
        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        output = output.transpose(0, 1) #output : [batch_size, seq_len, n_hidden * num_directions(=2)]

        attn_output, attention = self.attention_network(output,final_hidden_state)
        return attn_output, attention # attn_output : [batch_size, num_classes], attention : [batch_size, n_step]

#预测
def testing(model, x_test):
    # pred_data = model(x_test)
    x_test = x_test.to(device).to(torch.float32)
    pred, attention = model(x_test)
    return pred

#模型评估
def eva(pred_test,y_test):
    test_rmse = np.sqrt(np.mean(np.square(pred_test - y_test)))
    test_mae = np.mean(np.abs(pred_test-y_test))
    print("test_rmse:",test_rmse,"\n","test_mae:",test_mae)

#定义绘图函数
def fig(pred_test,y_test):
    # 绘图属性
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 绘图
    plt.figure()
    plt.plot(y_test, c='k', marker="*", label='实际值')
    plt.plot(pred_test, c='r', marker="o", label='预测值')
    plt.legend()
    plt.xlabel('样本点')
    plt.ylabel('y')
    plt.title('测试集对比')
    plt.savefig('./figure/测试集对比.jpg')
    plt.show()

# 绘图loss图
def figloss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('./figure/loss.jpg')
    plt.show()

def create_inout_sequences(input_data, input_window ,output_window):
    inout_seq = []
    L = len(input_data)
    block_len = input_window + output_window  # for one input-output pair
    block_num =  L - block_len + 1
    # total of [N - block_len + 1] blocks
    # where block_len = input_window + output_window

    for i in range(block_num):
        train_seq = input_data[i : i + input_window]
        # train_label = input_data[i + output_window : i + input_window + output_window]
        train_label = input_data[i + input_window : i + input_window + 1]
        inout_seq.append(np.append(train_seq, train_label))

    return np.array(inout_seq)

def BW_data_load():
    series = pd.read_csv('data/7Train1.csv', usecols=[0])

    # looks like normalizing input values curtial for the model
    bandwidth = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
    sampels = 3200
    train_data = bandwidth[:sampels] # 3200
    test_data = bandwidth[sampels:4200] # 1000

    # convert our train data into a pytorch train tensor
    train_sequence = create_inout_sequences(train_data, input_window, output_window)
    #test_data = torch.FloatTensor(test_data).view(-1)
    test_sequence = create_inout_sequences(test_data, input_window, output_window)
    x_train = train_sequence[:,0:input_window]
    y_train = train_sequence[:,input_window:input_window + 1]
    x_test = test_sequence[:,0:input_window]
    y_test = test_sequence[:,input_window:input_window + 1]

    #修改维度
    x_train = torch.tensor(x_train).unsqueeze(-1)
    y_train = torch.tensor(y_train).squeeze(0)
    x_test = torch.tensor(x_test).unsqueeze(-1)
    y_test = torch.tensor(y_test).squeeze(0)

    return x_train,y_train,x_test,y_test

#数据加载并标准化，返回数据
def data_load():
    data = pd.read_csv('data/datakun.csv').values
    row = data.shape[0]
    num_train = int(row * 0.8)  # 训练集

    # 训练集与测试集划分
    x_train = data[:num_train, 0:5]
    y_train = data[:num_train, 5:6]

    x_test = data[num_train:, 0:5]
    y_test = data[num_train:, 5:6]

    # 用样本数据拟合归一化器
    scaler1.fit(x_train)
    scaler2.fit(y_train)
    # 对训练数据进行归一化
    x_train = scaler1.transform(x_train)
    y_train = scaler2.transform(y_train.reshape(-1, 1))
    # 对测试数据进行归一化
    x_test = scaler1.transform(x_test)
    y_test = scaler2.transform(y_test.reshape(-1, 1))

    #修改维度
    x_train = torch.tensor(x_train).unsqueeze(-1)
    y_train = torch.tensor(y_train).squeeze(0)
    x_test = torch.tensor(x_test).unsqueeze(-1)
    y_test = torch.tensor(y_test).squeeze(0)

    return x_train,y_train,x_test,y_test

if __name__=='__main__':
    #加载数据
    x_train,y_train,x_test,y_test = BW_data_load()
    train_dataset = Data.TensorDataset(x_train, y_train)
    train_loader = Data.DataLoader(train_dataset, batch_size, True)
    test_dataset = Data.TensorDataset(x_test, y_test)
    test_loader = Data.DataLoader(test_dataset, batch_size, True)

    #训练模型
    # model,history = build_model(x_train,y_train,x_test,y_test)
    model = LSTM_Attention().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(500):
        for x, y in train_loader:
            x, y = x.to(device).to(torch.float32), y.to(device).to(torch.float32)
            pred, attention = model(x)
            loss = criterion(pred, y)
            if (epoch + 1) % 100 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #预测
    pred_test = testing(model, x_test)
    # 评估
    pred_test = pred_test.to('cpu', torch.double).detach().numpy()
    y_test = y_test.to('cpu', torch.double).detach().numpy()
    eva(pred_test, y_test)
    #反归一化
    y_test = scaler.inverse_transform(y_test)
    pred_test = scaler.inverse_transform(pred_test)
    #绘拟合图
    fig(pred_test, y_test)
    #绘loss图
    # figloss(history)

