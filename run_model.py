import numpy as np
import torch
from torch.utils.data import DataLoader

from model import fnn
from dataset import dds

N_EPOCHS = 200    
LEARNING_RATE = 0.005
BATCH_SIZE = 64
REPORT_EVERY = 1
input_n = 14
output_n = 1
HIDDEN_SIZE = 10

###### Add data
train_df = None
val_df = None
######

train_ds = dds(train_df)
val_ds = dds(val_df)

train_loader = DataLoader(dataset = train_ds, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(dataset = val_ds, batch_size = val_df.shape[0], shuffle = False)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = fnn(input_dim = input_n, out_dim = output_n, hidden_size = HIDDEN_SIZE).to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr=LEARNING_RATE)


# training
val_points = np.zeros((val_df.shape[0],2,N_EPOCHS//REPORT_EVERY))
loss_train_val = np.zeros((N_EPOCHS//REPORT_EVERY,2))
for epoch in range(N_EPOCHS):

    # start training
    model.train() 
    loss_epoch_train = 0
    for features, targets in train_loader:
        # forward propagation
        output = model(features)
        loss = loss_fn(output, targets)
        #initialize the gradient to zero
        optimizer.zero_grad()
        #back propagation
        loss.backward()
        #update the weights
        optimizer.step()
        
        loss_epoch_train+= targets.shape[0]*loss.item()
    
     
    # start eval
    model.eval()

    # predict validation set:
    feature_val, target_val = next(iter(val_loader)) # val_loader has only one batch
    prediction_val = model(feature_val)
    mse_val = loss_fn(prediction_val, target_val)

    
    if (epoch+1)%REPORT_EVERY == 0:   
       
        mse_epoch_train = loss_epoch_train/train_df.shape[0]
        print('Epoch {}/{}:'.format(epoch+1, N_EPOCHS))
        print('loss_train: {:4f}'.format(mse_epoch_train))
        print('loss_valid: {:4f}'.format(mse_val.item()))
        val_points[:, 0, (epoch+1)//REPORT_EVERY-1] = target_val[:,0].detach().numpy()
        val_points[:, 1, (epoch+1)//REPORT_EVERY-1] = prediction_val[:,0].detach().numpy()
        loss_train_val[int((epoch+1)/REPORT_EVERY)-1, 0] = np.round(mse_epoch_train, 4)
        loss_train_val[int((epoch+1)/REPORT_EVERY)-1, 1] = np.round(mse_val.item(), 4)