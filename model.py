#model definition
class Net(nn.Module):
    def __init__(self,input_size,ff_dim,n_head,n_classes,n_layers,dropout):
        super(Net, self).__init__()

        self.transformer_model = nn.Transformer(d_model=input_size, nhead=n_head, num_encoder_layers=n_layers,
                                                num_decoder_layers=n_layers, dim_feedforward=ff_dim, batch_first=True,dropout=dropout,device=device)
        self.fc = nn.Linear(64*9, n_classes)
        

    def forward(self, x):
        
        x = self.transformer_model(x,x)
        x = x.view(-1,64*9)
        x = self.fc(x)

        return x
