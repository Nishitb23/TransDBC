import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#load dataset
X_train, X_test, y_train, y_test = load_full_dataset()
print('Training dataset shape: ', X_train.shape, y_train.shape)
print('Testing dataset shape: ', X_test.shape, y_test.shape)

#converting data into tensor form and creating dataloader
X_train = torch.Tensor(X_train).to(device)
X_test = torch.Tensor(X_test).to(device)
y_train = torch.Tensor(y_train).reshape(-1).to(device)
y_test = torch.Tensor(y_test).reshape(-1).to(device)

train_dataset = TensorDataset(X_train,  y_train)
test_dataset = TensorDataset(X_test,  y_test)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

lossfn = nn.CrossEntropyLoss()

model = Net(input_size,ff_dim,n_head,n_classes,n_layers,dropout)
model.to(device)
optimizer= optim.Adam(model.parameters(), lr= learning_rate)

train(model,n_epochs,train_dataloader,optimizer,lossfn,test_dataloader)
