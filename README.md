# NCPVeReMi
A Privacy and Security Preserving Decentralized Federated Learining Based Liquin Neural Network for Misbehaviour Detection in Vehicle to Everything Communications.

## Background
 This repository was developed in conjunction with [this](linkToPaper) paper on the same topic created as part of an NSF REU at Kettering University. Propoesd in this work is a Decentralized Federated Learning method ideal for V2X communications, that utilizes a Closed-form Continuous-time based Neural Circuit Policy to effectively detect misbehaving vehicles in V2X networks by observing the data sent in Basic Safety Messages by a vehicle. This model aims to be resilient to model poisoning and data leakage by using secure decentralized federated learning. To learn more, [read our paper](Paper).

## Sourcing Data
The data for this project was generated as a part of [this](paperForVeReMiEx) paper by Josef Kamel, in which they extended the existing [VeReMi](link) dataset. Kamel's additions added many different simulated attacks, and expanded the information generated for each BSM. This VeReMi-Extension dataset was used for this paper, and processed with [this](CfCReadDatafromEXMultiThread.py) file. The processed files are availible on our google drive [here](googleDriveofData), as they are too large to be stored in our repository itself. Once dowloaded, place the CSV files in the /Data/ folder under the main folder of this repository for the code to access. 

## Prerequisites
To be able to run the code in this repository, you will need:
1. The capability to run Jupyter Notebook files
2. Python (Tested with 3.13.5)
3. The following packages:
    - [Neural Circuit Policies](https://ncps.readthedocs.io/en/latest/)
    - [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
    - [Numpy](https://numpy.org/)
    - [PyTorch](https://pytorch.org/)
    - [MatPlotLib](https://matplotlib.org/)

## Usage Instructions
There are two main files: [NCPModel](NCPModel.ipynb), which is the standard CfC NCP model created following the documentation of [this](CfCPaper) paper on using this model for misbehaviour detection, and [FederatedNCPModel](FederatedNCPModel.ipynb), which our federated learining implementations. In both of these files, the first several cells are importing the needed dependencies, defining the classes needed by the models, and formatting the datasets to be used by the different models. Run all cells above and including the OBU cell in order to setup everything properly. Once those cells have been run, the rest of the code can be used.

### NCPModel
For the [NCPModel](NCPModel.ipynb) file, the next cell you will see is the cell creating our OBU. This is where we set the parameters for the model, and decide if we are using the gpu. This file is designed to be as close to the CfC model defined in [this](CfCPaper) paper, in order to test its functionality before we expand on it. Because of this, refer to their paper for explanations of the parameters. 

After defining the test model, the next cell runs a test step on the model to ensure it is functioning properly, which was used during development, but is not necesary for operation. After that, we run a test before training to make sure the model is adequetely bad, and then train it. There are two cells that run the training, as one of them uses the full training dataset, which takes a while to run, and the other one runs a smaller subset of that dataset designed for quick training while we were testing. 

Finally, the last cell tests the model on the 20% of the dataset that we designated for testing. 


### FederatedNCPModel
This is where our novelty comes from. After running the beginning cells defining and contructing the dataset, the two cells right below that are two different federated learning methods.

The first method, labeled _Standard Federated Learning_, is an implementation of the standard, centralized federated learning, where there is a central server coordinating the learning. In V2X scenarios, this would most likely be an RSU running the training. There are many options in this cell for running: _deepTest_ runs a test on every iteration of every model, so you can track the improvement more closely. _weighing_ uses the loss of each training cycle to use the weights of models with less loss more, so idealy the models with a lower loss contribute more to the averaged model. _randomVehicles_ selects a different subset of the database to use every iteration, but results in each vehicle having different amounts of epochs being done on them. _doValidation_ runs a test on every individual model and on the global model in hopes of preventing model poisoning by only accepting models that perform better than the model currently on the vehicle. All of the further explanations for our parameters can be found in our paper, but what is currently in the repository are generally what we found to be most effective. 

Both methods in this paper output their results to separate text files, both placed in two different directories in the /out/ folder. For the standard federated learning, the file results contains the results of every test run, including a final test. The file Weights contains the aggregation weights of each model each iteration, which are calculated off of their loss. The Percs file contains the weights and percentage correct of each model after testing. As for the Decentralized model, the first file is the DeFedResults file, which contains the results of each model and the average result after the training is completed. The HistoricLoss file contains the loss of each model every epoch, and the SampleSizes file contains how many vehicles were sampled by each vehicle each epoch. The SelectedVehicles and SelectedWeights files contain the vehicles selected fro training by each model, and the probability of selection of each vehicle, respectively. Finally, the VehicleStatus file lists whether each vehicle is a malicious actor.

The Decentralized method, labeled _DeFTA: Decentralized Federated Training_, is based off of the work done in [this](patphapnh) paper on DeFTA (Decentralized Federated Trusted Averaging), which is a novel technique to securely train a machine learning model collaboratively without a central server. Again, more information about this can be found on our paper, but in general, the structure of this cell is the algorithm proposed by the paper ond DeFTA, modified to fit V2X communication's needs, and work properly with the CfC model. At the top, you will find the parameters for this model, including the number of vehicles in the simulation, max epochs, and most importantly, doEvil and percEvil, which decide if there are malitious model seeding vehicles, and how many. Running this cell run the DeFTA model, and a full simulation.

At the end of this file there are a couple cells very similar to the NCPModel file's cells, so that we could test an individual OBU with the federated learning modifications.

### Plotting
There is also a file Plotting included, which contains the code to plot all of the metrics of these models. This file needs to be run after running all of the :