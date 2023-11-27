# Ship Performance Model (NN-SPM)
NN-SPM estimates ship fuel consumption taking into consideration weather and operational conditions.


## Training Data
This repository contains 'example_dataset' that was built using:
1. Weather forecast data collected from NOAA for the experimental area in the North Pacific
2. Forecasts of FuelMassFlow generated from systems that are modelling ship energy system
    - Vessel is assumed to sail at constant speed, water depth, draft
    - For each weather combination, three different COGs are contained with the resulting FuelMassFlow estimate

Input features (n=9):
- wind speed (m/s)
- wind direction (degree)
- significant wave height (m)
- peak wave period (s)
- wave direction (degree)
- primary swell height (m)
- primary swell period (s)
- primary swell direction (degree)
- course over ground (degree)

Output (n=1): Fuel Mass Flow (kg/s)

## Model Architecture and Training Hyperparameters Values
A fully connected feedforward neural network.
- Layers: network has ten layers each with 10 neurons
- Activation function: Leaky ReLU
- Loss function : Mean Squared Error 
- Optimizer for loss function: ADAM
- Learning rate: 0.005
- Batch size: 64
- No. of Epochs: 30
