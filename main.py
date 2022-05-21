import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


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

#model training function
def train(model,n_epochs,trainloader,optimizer,lossfn,test_dataloader,onehotencoding=True,n_class=3):
  predlist=torch.zeros(0,dtype=torch.long, device='cpu')
  lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
  for epoch in range(n_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data   
      # zero the parameter gradients
      optimizer.zero_grad()
      # forward + backward + optimize
      #inputs = inputs.view(-1,1,64,9)
      outputs = model(inputs)
      #print(outputs.shape)
      if onehotencoding == True:
        #new_labels = torch.tensor(onehotvector[labels])
        #print(labels.shape)
        new_labels = torch.nn.functional.one_hot(labels.to(torch.int64),n_class).to(torch.float32)
      else:
        new_labels = labels
      loss = lossfn(outputs, new_labels)

      loss.backward()
      optimizer.step()
      # print statistics
      running_loss += loss.item()
      predlist=torch.cat([predlist,torch.argmax(outputs, dim=1).view(-1).cpu()])
      lbllist=torch.cat([lbllist,labels.view(-1).cpu()])
    print()
    print("The training loss for epoch ",epoch+1," is: ",loss)
    # Confusion matrix
    
    print("the training accuracy acheived is: ",accuracy_score(lbllist,predlist))
    test(model,test_dataloader,lossfn)
    
lossfn = nn.CrossEntropyLoss()

model = Net(input_size,ff_dim,n_head,n_classes,n_layers,dropout)
model.to(device)
optimizer= optim.Adam(model.parameters(), lr= learning_rate)

train(model,n_epochs,train_dataloader,optimizer,lossfn,test_dataloader)
