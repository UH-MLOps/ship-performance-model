# Ship Performance Model (NN-SPM)
NN-SPM estimates ship fuel consumption taking into consideration weather and operational conditions.

## Training Data
Built based on
1. Forecasts of fuel consumption generated from systems that are modelling ship energy system
2. Measurement data from sensors onboard a ship 

Input features:
- wind speed (m/s)
- wind direction (degree)
- significant wave height (m)
- peak wave period (s)
- wave direction (degree)
- primary swell height (m)
- primary swell period (s)
- primary swell direction (degree)
- course over ground (degree)
- speed over ground ()
- water depth
- current speed
- current direction

Output: Fuel Mass Estimation (kg/s)

## Model Architecture and Training Hyperparameters Values
A fully connected feedforward neural network.
- Layers: network has ten layers each with 10 neurons
- Activation function: Leaky ReLU
- Loss function : Mean Squared Error 
- Optimizer for loss function: ADAM
- Learning rate: 0.005
- Batch size: 64
- No. of Epochs: 30
