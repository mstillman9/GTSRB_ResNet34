This folder contains the code, dataset, and models

- GTSRB_Final_Training_Images.zip is the dataset
- assign7.py is the main code, containing the full training code
- dataset.py contains the dataset class
- model.py contains the model

For this assignment, I used a modified 34-layer ResNet. I have modified the last layer to a 43 class linear layer instead of 1000 classes. 
The produced models for each epoch are included in the following form:
modelXX-YY.YYYYYYYYYYYYYY.pth
with XX being the epoch number and YY.YYYYYYYYYYYYYY being the accuracy of the validation on that model.