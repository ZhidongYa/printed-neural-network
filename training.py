import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import calendar
import time

def OneHot(y, K):
    M = y.shape[1]
    Y = np.zeros([K, M])
    for i in range(M):
        Y[int(y[0, i]), i] = 1
    return Y

def GenerateData(N, var, plot, seed):
    random.seed(seed);
    np.random.seed(seed);
    x1 = np.random.randn(1, N) * var + 1
    y1 = np.random.randn(1, N) * var + 1
    l1 = np.random.randn(1, N) * 0 + 0
    x2 = np.random.randn(1, N) * var + 1
    y2 = np.random.randn(1, N) * var - 1
    l2 = np.random.randn(1, N) * 0 + 1
    x3 = np.random.randn(1, N) * var - 1
    y3 = np.random.randn(1, N) * var + 1
    l3 = np.random.randn(1, N) * 0 + 2
    x4 = np.random.randn(1, N) * var - 1
    y4 = np.random.randn(1, N) * var - 1
    l4 = np.random.randn(1, N) * 0 + 3
    X1 = np.hstack((x1, x2, x3, x4))
    X2 = np.hstack((y1, y2, y3, y4))
    X = np.vstack((X1, X2))
    y = np.hstack((l1, l2, l3, l4))
    Y = OneHot(y, 4)
    if plot:
        plt.figure()
        plt.scatter(X[0, :], X[1, :], c=y)
        plt.show()
    return X.astype(float), y.astype(np.int64)


def SplitData(X,Y, training_rate,valid_rate,seed=0):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    N, M = X.shape
    K = Y.shape[0]
    M_train = int(M * training_rate)
    M_valid = int(M * (valid_rate+training_rate))
    DATA = torch.cat((X, Y),dim=0)
    #index = np.arange(0, M, 1)
    #np.random.shuffle(index)
    index=torch.randperm(M)
    DATA = DATA[:, index]
    X = DATA[:-K, :].reshape(N, -1)
    Y = DATA[-K:, :].reshape(K, -1)
    X_train = X[:, :M_train]
    Y_train = Y[:, :M_train]
    X_valid = X[:, M_train:M_valid]
    Y_valid = Y[:, M_train:M_valid]
    X_test = X[:, M_valid:]
    Y_test = Y[:, M_valid:]
    return X_train, Y_train, X_valid, Y_valid,X_test,Y_test

def Normalization(X):
    return (X - np.mean(X, 1).reshape(-1, 1)) / np.std(X, 1).reshape(-1, 1)

def train_nn_with_scheduler(epoch_times,NN, train_loader, valid_loader, optimizer, lossfunction,Epoch=10**10):
    training_ID = ts = int(calendar.timegm(time.gmtime()))
    print(f'The ID for this training is {training_ID}.')
    
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    best_valid_loss = 100000
    not_decrease = 0
    
    for epoch in range(Epoch):
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            prediction = NN(x_train)
            loss = lossfunction(prediction, y_train)
            
            yhat = torch.argmax(prediction.data, 1)
            train_correct = torch.sum(yhat == y_train.data)
            acc_train = train_correct / y_train.numel()
            train_acc.append(acc_train)
            
            loss.backward()
            optimizer.step()
        #scheduler.step()
        train_loss.append(loss.data)
        
        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                prediction_valid = NN(x_valid)
                loss_valid = lossfunction(prediction_valid, y_valid)
                
                yhat_valid = torch.argmax(prediction_valid.data, 1)
                valid_correct = torch.sum(yhat_valid == y_valid.data)
                acc_valid = valid_correct / y_valid.numel()
                valid_acc.append(acc_valid)

        if loss_valid.data < best_valid_loss:
            best_valid_loss = loss_valid.data
            torch.save(NN, f'./temp/NN_scheduler_temp_{training_ID}')
            not_decrease = 0
        else:
            not_decrease += 1
        
        valid_loss.append(loss_valid.data)
        
        if not_decrease > 500:
            print('Early stop.')
            epoch_times.append(epoch)
            
            break
            
        if not epoch % 500:
            print(f'| Epoch: {epoch:-8d} | Valid accuracy: {acc_valid:.5f} | Valid loss: {loss_valid.data:.9f} |')
            
    print('Finished.')
    return torch.load(f'./temp/NN_scheduler_temp_{training_ID}'), train_loss, valid_loss, train_acc, valid_acc,epoch_times
