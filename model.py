import torch

class fnn(torch.nn.Module):
    def __init__(self, input_dim, out_dim, hidden_size):
        super().__init__()
        self.inputs = input_dim
        self.outputs = out_dim
        self.hidden = hidden_size
        
        # layers
        self.fc1 = torch.nn.Linear(self.inputs, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc5 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc6 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc7 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc8 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc9 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc10 = torch.nn.Linear(hidden_size, self.outputs)
        self.bn = torch.nn.BatchNorm1d(hidden_size)

        
        # activation
        self.lrelu = torch.nn.LeakyReLU(0.2)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.elu = torch.nn.ELU()
        self.sigmoid= torch.nn.Sigmoid()
        self.logsigmoid= torch.nn.LogSigmoid()
        
        # initialize the weights
        self.init_weights()
        
    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)
        torch.nn.init.kaiming_normal_(self.fc6.weight)
        torch.nn.init.kaiming_normal_(self.fc7.weight)
        torch.nn.init.kaiming_normal_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.fc9.weight)
        torch.nn.init.kaiming_normal_(self.fc10.weight)


        
    def forward(self, features):
        output = self.fc1(features)
        output = self.lrelu(output)  
        output = self.fc2(output)
        output = self.lrelu(output)
        output = self.fc3(output)
        output = self.lrelu(output)
        output = self.fc4(output)
        output = self.lrelu(output)
        output = self.fc5(output)
        output = self.lrelu(output)       
        output = self.fc6(output)
        output = self.lrelu(output)
        output = self.fc7(output)
        output = self.lrelu(output)
        output = self.fc8(output)
        output = self.lrelu(output)
        output = self.fc9(output)

        output = self.lrelu(output)
        output = self.fc10(output)
        return output