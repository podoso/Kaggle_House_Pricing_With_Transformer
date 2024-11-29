import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from my_model import My_Transformer
import pandas as pd
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
# ### Import Data
train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

# ##### description
# print(train_raw.shape, test_raw.shape) # (1460, 81) (1459, 80)
# print(train_raw.iloc[0:4,[0,1,2,3,-3,-2,-1]])

# ##### Concatenate the train and test data for standardisation
all_features_raw = pd.concat((train_raw.iloc[:,1:-1], test_raw.iloc[:,1:]))
all_features = all_features_raw.copy()

# ### Data Preprocessing
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x-x.mean())/(x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

all_features = pd.get_dummies(all_features, dummy_na=True, dtype=int)
# print(all_features.shape)
# print(all_features)

seq_length = 1
batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_train = train_raw.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32).to(device)
train_features = train_features.view(-1, seq_length, train_features.shape[1])
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32).to(device)
test_features = test_features.view(-1, seq_length, test_features.shape[1])
train_labels = torch.tensor(train_raw.SalePrice.values.reshape(-1,1), dtype=torch.float32).to(device)
train_labels = train_labels.view(-1, seq_length, train_labels.shape[1]) # torch.Size([1460, 1, 1])
print("The shape of train labels is: ", train_labels.shape, )
src_data = train_features
tgt_data = train_labels
expanded_data1 = tgt_data.repeat(1, 1, 330)
expanded_data2 = tgt_data.tile(1, 1, 330)
tgt_data = expanded_data1 # torch.Size([1460, 1, 330])

dataset = CustomDataset(src_data, train_labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
my_transformer = My_Transformer(d_model=330, nhead=6,
                              num_encoder_layers=6, num_decoder_layers=6)

# 不需要做mask,因为进行训练的数据不包含序列信息
src_mask = torch.zeros((16, 1, 330)).to(device)
tgt_mask = torch.zeros((16, 1, 330)).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(my_transformer.parameters())

# Log root mean squared error
def log_rmse(predictions, labels):
    # print("The features shape is: ",predictions.shape)
    # print("The labels shape is: ",labels.shape)
    assert predictions.shape == labels.shape
    clipped_preds = torch.clamp(predictions, 1, float('inf'))
    rmse = torch.sqrt(criterion(torch.log(clipped_preds), torch.log(labels)))
    return rmse

# K-fold cross-validation
def My_get_k_fold_data(k, i, X, y_d_demension, y1):
    """
    params: 
    k: number of folds
    i: index of the fold
    X: src_data
    y1: train_labels, the last dimension is 1 
    y_d_demension: the dimension of the labels, e.g., 330
    return X_train, y_train1, y_train2, X_valid, y_valid1, y_valid2
    y_train1: labels of the train set, last dimension is d_model
    y_train2: labels of the train set, last dimension is 1
    y_valid1: labels of the validation set, last dimension is d_model
    y_valid2: labels of the validation set, last dimension is 1
    """
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train1, y_train2 = None, None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y1_part, y_d_demension_part = X[idx, :], y1[idx, :], y_d_demension[idx, :]
        if j == i:
            X_valid, y_valid1, y_valid2 = X_part, y1_part, y_d_demension_part

        elif X_train is None:
            X_train, y_train1, y_train2 = X_part, y1_part, y_d_demension_part
            
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train1 = torch.cat([y_train1, y1_part], 0)
            y_train2 = torch.cat([y_train2, y_d_demension_part], 0)

    return X_train, y_train1, y_train2, X_valid, y_valid1, y_valid2

def k_fold(k, X_train, y_train_1, y_train_d_demension, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    train_losses, valid_losses = [], []
    for i in range(k):
        data = My_get_k_fold_data(k, i, X_train, y_train_1, y_train_d_demension)
        net = My_Transformer(d_model=330, nhead=6, num_encoder_layers=6, num_decoder_layers=6).to(device)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        train_losses.append(train_ls)
        valid_losses.append(valid_ls)
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k, train_losses, valid_losses

def train(net, train_features, train_labels_d_dimension, train_labels_1, valid_features, valid_labels_d_demension, valid_labels_1, num_epochs, learning_rate, weight_decay, batch_size):
    # train iterators
    train_iter1 = DataLoader(CustomDataset(train_features, train_labels_d_dimension), batch_size=batch_size, shuffle=True)
    train_iter2 = DataLoader(CustomDataset(train_features, train_labels_1), batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_losses, valid_losses = [], []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for (src1, tgt1),(_, tgt2) in zip(train_iter1,train_iter2):
            optimizer.zero_grad()
            output = net(src1, tgt1)
            # print("The shape of output is: ", output.shape)
            assert output.shape == tgt2.shape
            loss = log_rmse(output, tgt2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_iter2))
        if valid_features is not None and valid_labels_d_demension is not None:
            out = net(valid_features, valid_labels_d_demension)
            # print("The shape of valid out is: ", out.shape)
            assert out.shape == valid_labels_1.shape
            valid_loss = log_rmse(out, valid_labels_1)
            valid_losses.append(valid_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Valid Loss: {valid_losses[-1]:.4f}' if valid_features is not None else f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}')
    return train_losses, valid_losses

# Train the transformer model
k, num_epochs, lr, weight_decay, batch_size = 5, 10, 0.80, 0, 16
train_l, valid_l, train_losses, valid_losses = k_fold(k, src_data, train_labels, tgt_data, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-fold validation: avg train log rmse: {float(train_l):f}, avg valid log rmse: {float(valid_l):f}')

# plot the loss curve
plt.figure(figsize=(10, 5))
for i in range(k):
    plt.plot(range(1, num_epochs + 1), train_losses[i], label=f'Train Fold {i+1}')
    plt.plot(range(1, num_epochs + 1), [loss.detach().cpu().numpy() for loss in valid_losses[i]], label=f'Valid Fold {i+1}', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Log RMSE')
plt.legend()
plt.title('Training and Validation Log RMSE over Epochs')
plt.show()