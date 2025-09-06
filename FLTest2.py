# Imports
from ncps.wirings import AutoNCP
from ncps.torch import CfC
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from numpy import genfromtxt
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.nn as nn
from random import sample
import os
import time
import csv
import sys
import json

torch.set_float32_matmul_precision("high")

batchSize = 64

# FORMATTING DATASET FOR FED. LEARNING
testName = 'RandPos'
doEvil = False
percEvil = 20
dataFile = 'Data/CfCMultiExtension/RandomPos_0709.csv'
dataSet = genfromtxt(dataFile, delimiter=',')
dataSet = np.delete(dataSet, 0, axis=0) # Remove the labels at the beginning of the dataset

# Devide dataset into reciever groups
fedDataSet = dataSet[np.argsort(dataSet[:, 1])] # Sort dataset by reciver ID
_, counts = np.unique(fedDataSet[:,1], return_counts=True) # Get the indexes of the change in datasets.
sum = 0
for i in range(len(counts)): # Accumulating counts so that we can use them as indexes
    sum += counts[i]
    counts[i] = sum
fedDataSet = np.split(fedDataSet, counts) # Split larger dataset into per vehicle datasets.

newData = [] 
for reciever in fedDataSet: # Go through all vehicle datasets
    subData = []
    index = 0
    while index < len(reciever) - 10: # organize the new dataset as a list of chuncks of 10 messages 
        subData.append(reciever[index:index+10])
        index += 5
    subData = torch.Tensor(subData)
    if subData.shape[0] != 0:
        newData.append(subData) # Create tensor from per vehicle dataset and add to list of datas.
fedDataSet = newData
# Final output of this cell is fedDataSet, a list of the datasets of each vehicle.


#PROPER FORMATTING FOR TESTING DATASETS
#Time sequences are 10 timepoints (Messages) with 7 features per message.
#Organized by car.
unq, counts = np.unique(dataSet[:, 2], return_counts = True)
sender = 0
lastSenderCount = 0
newData = []
# Organize dataset into sets of 10 messages by sender
while sender < counts.shape[0]:
    # Loop through sender
    index = 0
    while index < counts[sender] - 10:
        # Loop through messages from sender
        newData.append(dataSet[lastSenderCount+index:lastSenderCount +index+10])
        index += 5
    sender += 1
    lastSenderCount += counts[sender-1]
dataSet = torch.tensor(newData)
leng = dataSet.shape[0]
trainPerc = 80
# Create seperate datasets for testing and training, using Train Percentage as metric for split
trainDataIn = torch.Tensor(dataSet[:int(leng*(trainPerc/100)),:,3:10]).float()
trainDataOut = torch.Tensor(np.int_(dataSet[:int(leng*(trainPerc/100)),:,11])).long()
testDataIn = torch.Tensor(dataSet[int(leng*(trainPerc/100)):,:,3:10]).float()
testDataOut = torch.Tensor(np.int_(dataSet[int(leng*(trainPerc/100)):,:,11])).long()
newsetIn = []
newsetOut = []
testsetIn = []
testsetOut = []
tinyTestIn = []
tinyTestOut = []
# Create tiny dataset to run verification tests
for index in range(leng):
    if not (int(index/10) % 300):
        tinyTestIn.append(dataSet[index, :, 3:10])
        tinyTestOut.append(dataSet[index, :, 11])
# Create dataset of 1/100th of the entries for quicker testing during development
for index in range(0,int((leng) * (trainPerc/100))):
    if not (int(index/10) % 100):
        newsetIn.append(dataSet[index,:,3:10])
        newsetOut.append((dataSet[index,:,11]))
for idx in range(int((leng) * (trainPerc/100)), leng):
    if not (int(idx/10) % 10):
        testsetIn.append(dataSet[idx,:,3:10])
        testsetOut.append((dataSet[idx,:,11]))
testingIn = torch.Tensor(np.array(newsetIn)).float()
testingOut = torch.Tensor(np.array(newsetOut)).long()
inTest = torch.Tensor(np.array(testsetIn)).float()
outTest = torch.Tensor(np.array(testsetOut)).long()
tinyTestIn = torch.Tensor(np.array(tinyTestIn)).float()
tinyTestOut = torch.Tensor(np.array(tinyTestOut)).long()
# Create Dataloaders for all the datasets
dataLoaderTrain = data.DataLoader(data.TensorDataset(trainDataIn, trainDataOut), batch_size=batchSize, shuffle=False, num_workers=10, persistent_workers = True, drop_last= True)
dataLoaderTest = data.DataLoader(data.TensorDataset(testDataIn, testDataOut), batch_size=batchSize, shuffle=False, num_workers=10, persistent_workers = True, drop_last= True)
testingDataLoader = data.DataLoader(data.TensorDataset(testingIn, testingOut), batch_size=batchSize, shuffle = False, num_workers=10, persistent_workers = True, drop_last= True)
testingTestData = data.DataLoader(data.TensorDataset(testingIn, testingOut), batch_size=batchSize, shuffle = False, num_workers=10, persistent_workers = True, drop_last= True)
tinyTestLoader = data.DataLoader(data.TensorDataset(tinyTestIn, tinyTestOut), batch_size=batchSize, shuffle = False, num_workers = 10, persistent_workers= True)


# Creating Learner
class CfCLearner(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lossFunc = nn.CrossEntropyLoss()
        self.loss = None
    
    def training_step(self, batch, batch_idx):
        # Get in and out from batch
        inputs, target = batch
        # Put input through model
        output, _ = self.model.forward(inputs)
        # Reorganize inputs for use with loss function
        output = output.permute(0, 2, 1)
        # Calculate Loss using Cross Entropy Loss 
        loss = self.lossFunc(output, target)
        self.log("trainLoss", loss, prog_bar=True)
        self.loss = loss
        return loss

    def validation_step(self, batch, batch_idx):
        # Get in and out from batch
        inputs, target = batch
        # Put input through model
        output, _ = self.model.forward(inputs)
        # Reorganize inputs for use with loss function
        output = output.permute(0, 2, 1)
        print(f"output: {output.shape}")
        print(f"target: {target.shape}")
        # Calculate Loss using Cross Entropy Loss 
        loss = self.lossFunc(output, target)
        self.log("valLoss", loss, prog_bar=True)
        self.loss = loss
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        # Using AdamW optomizer based on info from paper
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr = 0.001)
        # return ([optimizer], [torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.6)])
        return torch.optim.AdamW(self.model.parameters(), lr = 0.01) # TESTING REMOVING THE SCHEDULER
    

    # Creating Model/Module
class Modena(nn.Module): 
    # CfC with feed-forward layer to classify at end.
    def __init__(self, inputSize, unitNum = None, motorNum = 2, outputDim = 2, batchFirst = True):
        super().__init__()
        #Allow for creation of a copy of another instance
        if isinstance(inputSize, Modena):
            self.inputSize = inputSize.inputSize
            self.unitNum = inputSize.unitNum
            self.motorNum = inputSize.motorNum
            self.outputDim = inputSize.outputDim
            self.batchFirst = inputSize.batchFirst
            # Create NCP wiring for CfC
            wiring = AutoNCP(self.unitNum, self.motorNum)
            # Create CfC model with inputs and wiring
            self.cfc = CfC(self.inputSize, wiring, batch_first=self.batchFirst)
            # Create feed-forward layer
            self.fF = nn.Linear(self.motorNum, self.outputDim)
            self.fF.weight = nn.Parameter(inputSize.fF.weight)
        else:
            self.inputSize = inputSize
            self.unitNum = unitNum
            self.motorNum = motorNum
            self.outputDim = outputDim
            self.batchFirst = batchFirst
            # Create NCP wiring for CfC
            wiring = AutoNCP(unitNum, motorNum)
            # Create CfC model with inputs and wiring
            self.cfc = CfC(inputSize, wiring, batch_first=batchFirst)
            # Create feed-forward layer
            self.fF = nn.Linear(motorNum, outputDim)
        

    def forward(self, batch, hidden = None):
        batch, hidden = self.cfc(batch, hidden) # Pass inputs through CfC
        out = nn.functional.relu(self.fF(batch)) # pass through FeedForward Layer, then make 0 minimum
        return out, hidden # Return the guess and the hidden state
    

# Creating overall model Class
class OBU():
    def __init__(self, inputSize, units = 20, motors = 8, outputs = 20, epochs = 10, lr = 0.01, randInt = 0, gpu = False, dataset = None, evil = False):
        if isinstance(inputSize, OBU):
            self.lr = inputSize.lr
            self.epochs = inputSize.epochs
            self.gpu = inputSize.gpu
            self.model = Modena(inputSize.model)
            self.model.load_state_dict(inputSize.model.state_dict())
            self.learner = CfCLearner(self.model, self.lr)
            self.learner.load_state_dict(inputSize.learner.state_dict())
            self.trainer = pl.Trainer(
                logger = CSVLogger('log/Fed'), # Set ouput destination of logs, logging accuracy every 50 steps
                max_epochs = self.epochs, # Number of epochs to train for
                gradient_clip_val = 1, # This is said to stabilize training, but we should test if that is true
                accelerator = "gpu" if self.gpu else "cpu" # Using the GPU to run training or not
                )

        else:
            self.lr = lr
            self.epochs = epochs
            self.gpu = gpu
            self.model = Modena(inputSize, units, motors, outputs)
            self.learner = CfCLearner(self.model, lr) # tune units, lr
            self.trainer = pl.Trainer(
                logger = CSVLogger('log/Fed'), # Set ouput destination of logs, logging accuracy every 50 steps
                max_epochs = epochs, # Number of epochs to train for
                gradient_clip_val = 1, # This is said to stabilize training, but we should test if that is true
                accelerator = "gpu" if gpu else "cpu" # Using the GPU to run training or not
                )
        # Creating variables needed for DeFL
        self.prevAccuracy = 0
        self.evil = evil
        self.nearbyOBUs = []
        self.id = None
        self.dataset = dataset
        self.outnum = 0
        self.confidences = []
        self.samplingWeights = []
        self.priority = 0
        self.datalen = 0
        self.otherPriorities = []
        self.sampling = []
        self.curr_loss = 0
        self.prev_loss = None
        self.trust_loss = 0
        self.curr_acc = 0
        self.prev_acc = None
        self.trust_acc = 0
        self.backupWeights = self.learner.state_dict()
        self.prevWeights = dict(self.learner.state_dict())
        self.goodNeighbors = []
        self.testing = True
        self.toTest = []
        self.perEpoch = 30
        self.rounds = 0
        self.curr_f1 = 0
        self.prev_f1 = None
    

    # Overloading add function to create fed.avg. model
    def __add__(self, other):
        self.learner.load_state_dict(dict( (n, self.learner.state_dict().get(n, 0)+other.learner.state_dict().get(n, 0)) for n in set(self.learner.state_dict())|set(other.learner.state_dict()) ))
        return self
    
    # Overloading multiplication function to add weights
    def __mul__(self, i):
        self.learner.load_state_dict(dict((n, self.learner.state_dict().get(n, 0)*i) for n in self.learner.state_dict()))
        return self

    # Overloading div. function to average model
    def __truediv__(self, i):
        self.learner.load_state_dict(dict((n, self.learner.state_dict().get(n, 0)/i) for n in self.learner.state_dict()))
        return self
    
    def fit(self, dataLoader):
        # calling built in fit function
        self.trainer.fit(self.learner, dataLoader) 
        return self.learner.loss
    
    # Function to run model through a testing dataset and calculate accuracy. Can be expanded to give more metrics and more useful metrics.
    def test(self, dataIn, dataOut, mathy = False):
        # Put input data through model and determine classification
        with torch.no_grad():
            outs = np.asarray(self.model(dataIn)[0])
        outs = torch.from_numpy(outs)
        # Get the label with the maximum confidence for determining classification
        print(outs.shape)
        _, res = torch.max(outs, 2)
        Pt = Pf = Nt = Nf = 0
        countR = 0
        numZero = 0
        tot = outs.shape[0]
        total = 0
        for i in range(0, tot):
            # Loop through sequences of 10 each
            for t in range(0, res[i].shape[0]):
                # Loop through the sub-sequences
                if res[i,t] == dataOut[i,t]:
                    if res[i,t] == 0:
                        Nt += 1
                        numZero += 1
                    else:
                        Pt += 1
                    # Check if label is correct, and add to count right accordingly
                    countR += 1
                else:
                    if dataOut[i,t] == 0:
                        Pf += 1
                        numZero += 1
                    else:
                        Nf += 1
                total += 1
        # Mathy determines if we want the program to output just the numbers or a string for easy readability
        if mathy:
            # If we have at least one true positive, we can do all the other calculations.
            if Pt != 0:
                # Calculate accuracy, precision, recall and f1.
                accuracy = (Pt+Nt)/(Pt+Pf+Nf+Nt)
                precision = (Pt)/(Pt+Pf)
                recall = (Pt)/(Pt+Nf)
                f1 = (2*precision*recall)/(precision+recall)
                print(precision)
                print(recall)
                print("Model got " + str(countR) + "/" + str(total) + " right.")
                print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
                print(f"{numZero}, {numZero/total * 100}% Zeroes, {total-numZero} Non Zero entries.")
                return f1, recall, precision, accuracy
            else:
                # At least calculate accuracy when the model predicts zero.
                accuracy = (Pt+Nt)/(Pt+Pf+Nf+Nt)
                print("Model could not complete tests.")
                return 0, 0, 0, accuracy 
        else:
            # If we have at least one true positive, we can do all the other calculations.
            if Pt != 0:
                # Calculate accuracy, prcecision, recall and f1.
                accuracy = (Pt+Nt)/(Pt+Pf+Nf+Nt)
                precision = (Pt)/(Pt+Pf)
                recall = (Pt)/(Pt+Nf)
                f1 = (2*precision*recall)/(precision+recall)
                print(precision)
                print(recall)
                print("Model got " + str(countR) + "/" + str(total) + " right.")
                print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
                print(f"{numZero}, {numZero/total * 100}% Zeroes, {total-numZero} Non Zero entries.")
                return f"Model got {countR}/{total} right. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}"
            else:
                print("Model could not complete tests.")
                return f"Model could not complete tests, found 0 of misbehaviour."
    
    # Run a test step of the model.
    def testStep(self, dataLoader):
        self.learner.validation_step(next(iter(dataLoader)), 0)
    
    # Re-define the model.
    def setModel(self, model):
        if not model == None:
            self.model = model

    # Return the current model.
    def getModel(self):
        return self.model
    
    # Get the saved vehicle weights.
    def getSavedState(self):
        return self.prevWeights

    # Update the saved vehicle weights.
    def updateSavedStates(self):
        # If this OBU is acting as malicious, we return the incorrect weights.
        if self.evil:
            self.prevWeights = dict((n, torch.full(self.learner.state_dict()[n].shape,10000000)) for n in self.learner.state_dict()).copy()
            return # Never update weights, so always passing on very large weights
        self.prevWeights = self.learner.state_dict().copy()
    
    # get the current state.
    def getState(self):
        return self.learner.state_dict()
    
    # Restore the OBU from the saved state.
    def restoreFromBackup(self):
        self.trainer.fit_loop.max_epochs = self.trainer.current_epoch - self.perEpoch
        self.trainer.fit(self.learner, self.dataset, ckpt_path=f'log/Fed/{self.id}checkpoint.ckpt')

    # Save the current state as a backup.
    def saveBackup(self):
        self.trainer.save_checkpoint(f'log/Fed/{self.id}checkpoint.ckpt')
    
    # Get malicious status of vehicle.
    def isEvil(self):
        return True if self.evil else False
    
    # Set state of learner and model. One input just sets the state, and two inputs adds them together first.
    def setState(self, one, two = None):
        if two:
            tom = dict((n, one.get(n, 0)+two.get(n, 0)) for n in set(one)|set(two))
        else:
            tom = one
        self.learner.load_state_dict(tom)
        return tom
    
    # Run one sub-epoch of the model.
    def step(self, epochs):
        self.perEpoch = epochs
        self.trainer.fit_loop.max_epochs = self.trainer.current_epoch + epochs
        self.curr_loss = self.fit(self.dataset).item()
        return self.curr_loss

    # Update the vehicles we are sampling for this epoch.
    # Not used by DeFL/DeFTA anymore.
    def updateSelected(self):
        self.sampling = []
        count = 0
        for idx in self.nearbyOBUs:
            rand = np.random.randint(0, 100)
            if rand <= int(100*self.samplingWeights[idx]): # Less than or equal, as we want 1 to be selected every time.
                self.sampling.append(int(idx))
                count += 1
        return self.sampling # Returning how many vehicles were selected for training
        
    # Reset the trainer with a new trainer.
    def resetTrainer(self):
        self.trainer = pl.Trainer(
            logger = CSVLogger('log'), # Set ouput destination of logs, logging accuracy every 50 steps
            max_epochs = self.epochs, # Number of epochs to train for
            gradient_clip_val = 1, # This is said to stabilize training, but we should test if that is true
            accelerator = "gpu" if self.gpu else "cpu" # Using the GPU to run training or not
            )
        

# Define class to monitor all metrics of model to generate graphs and compare.
class OutLogger():
    def __init__(self, path):
        #Helpers
        self.path = path
        self.epochTimes = []
        self.times = []

        #Outs
        self.avgLossVEpoch = []
        self.avgF1VEpoch = []
        self.avgRecallVEpoch = []
        self.avgPrecisionVEpoch = []
        self.avgAccuracyVEpoch = []
        self.lossVPercEvil = None
        self.F1VPercEvil = None
        self.RecallVPercEvil = None
        self.PrecisionVPercEvil = None
        self.AccuracyVPercEvil = None
        self.AvgVehicleTime = None
        self.MaxVehicleTime = None
        self.TotTime = None

    # Define start and end for timing the per vehicle time.
    def startVehicleTimer(self):
        self.startTime = time.time()
    def endVehicleTimer(self):
        self.times.append(time.time()-self.startTime)

    # Define start and end for timing the full epoch time.
    def startEpochTimer(self):
        self.startEpochTime = time.time()
    def endEpochTimer(self):
        self.epochTimes.append(time.time()-self.startEpochTime)

    # Update the logs by running a test per vehicle. 
    def updateLogs(self, vehicles, epoch):
        currLoss = 0
        currF1 = 0
        currRecall = 0
        currPrecision = 0
        currAccuracy = 0
        count = 0
        for vehicle in vehicles:
            currLoss += vehicle.curr_loss
            f1, recall, precision, accuracy = vehicle.test(testingIn, testingOut, True)
            currF1 += f1
            currRecall += recall
            currPrecision += precision
            currAccuracy += accuracy
            count += 1
        self.avgLossVEpoch.append([epoch, currLoss/count])
        self.avgF1VEpoch.append([epoch, currF1/count])
        self.avgRecallVEpoch.append([epoch, currRecall/count])
        self.avgPrecisionVEpoch.append([epoch, currPrecision/count])
        self.avgAccuracyVEpoch.append([epoch, currAccuracy/count])
            
    # Add the final times and values to their respective point in the log.
    def finalLogs(self, vehicles, percEvil):
        self.lossVPercEvil = [percEvil, self.avgLossVEpoch[-1][1]]
        self.F1VPercEvil = [percEvil, self.avgF1VEpoch[-1][1]]
        self.RecallVPercEvil = [percEvil, self.avgRecallVEpoch[-1][1]]
        self.PrecisionVPercEvil = [percEvil, self.avgPrecisionVEpoch[-1][1]]
        self.AccuracyVPercEvil = [percEvil, self.avgAccuracyVEpoch[-1][1]]
        self.AvgVehicleTime = np.sum(self.times)/len(self.times)
        self.MaxVehicleTime = np.max(self.times)
        self.TotTime = np.sum(self.epochTimes)/len(self.epochTimes)

    # Ouput the logged values to files.
    def log(self):
        path = f"out/{self.path}"
        if not os.path.exists(f"out/{self.path}"):
            os.makedirs(f"out/{self.path}")
        with open(f'{path}avgLossVEpoch.csv', 'w', newline='') as filename:
            writer = csv.writer(filename)
            writer.writerow(['epoch', 'avg Loss'])
            writer.writerows(self.avgLossVEpoch)
        with open(f'{path}avgF1VEpoch.csv', 'w', newline='') as filename:
            writer = csv.writer(filename)
            writer.writerow(['epoch', 'avg F1'])
            writer.writerows(self.avgF1VEpoch)
        with open(f'{path}avgRecallVEpoch.csv', 'w', newline='') as filename:
            writer = csv.writer(filename)
            writer.writerow(['epoch', 'avg Recall'])
            writer.writerows(self.avgRecallVEpoch)
        with open(f'{path}avgPrecisionVEpoch.csv', 'w', newline='') as filename:
            writer = csv.writer(filename)
            writer.writerow(['epoch', 'avg Precision'])
            writer.writerows(self.avgPrecisionVEpoch)
        with open(f'{path}avgAccuracyVEpoch.csv', 'w', newline='') as filename:
            writer = csv.writer(filename)
            writer.writerow(['epoch', 'avg Accuracy'])
            writer.writerows(self.avgAccuracyVEpoch)
        others = {'Loss V PercEvil':self.lossVPercEvil, 'F1 V PercEvil':self.F1VPercEvil, 'Recall V PercEvil':self.RecallVPercEvil, 'Precision V PercEvil':self.PrecisionVPercEvil, 
                  'Accuracy V PercEvil':self.AccuracyVPercEvil, 'Max Per-Vehicle Time':self.MaxVehicleTime, 'Avg Per-Vehicle Time':self.AvgVehicleTime, 'Total Time Per Epoch':self.TotTime}
        with open(f'{path}avgAccuracyVEpoch.csv', 'w', newline='') as filename:
            writer = csv.writer(filename)
            writer.writerow(['epoch', 'avg Accuracy'])
            writer.writerows(self.avgAccuracyVEpoch)
        with open(f'{path}ExtraData.json', 'w') as filename:
            json.dump(others, filename)



# Standard Federated Learning

# With 2 epochs, 10 sub, and 0:3 models ending acc. of 0.1377%
# With 2 epochs, 10 sub, and 0:10 models ending acc. of

# Tested with weights, but weighing off of the loss leads to choosing the models trained by vehicles without attacks.
pl.seed_everything(1000)
results = {}
dataSets = {}
models = {}
histWeights = []
percentages = []
cars = []
rcvrs = []
percs = {}
states = {}
subEpochs = 10 # 50
epochs = 60 # 5
vehicles = 200 # 50
lr = 0.01
motors = 8
units = 20
batchSize = 64
gpu = False
deepTest = False
weighing = False
randomVehicles = False
doValidation = False
avgLossVEpoch = []
avgF1VEpoch = []
avgRecallVEpoch = []
avgPrecisionVEpoch = []

# Create starting models
mainModel = OBU(7, epochs= subEpochs, gpu = gpu, lr = lr, motors = motors, units = units)
nextModel = OBU(7, epochs= subEpochs, gpu = gpu, lr = lr, motors = motors, units = units)
path = f"FL/{testName}-{doEvil}-{percEvil}-{epochs}-{subEpochs}-{vehicles}/"
if not os.path.exists(f"out/{path}"):
    os.makedirs(f"out/{path}")

log = OutLogger(path)

if not randomVehicles:
    # Divide dataset of recieving vehicles among OBUs
    rcvrs = []
    for vehicle in fedDataSet[500:vehicles+500]: # 10
        rcvrID = int(vehicle[0,0,2].item())
        # Add new OBU for each model
        if np.random.randint(0,100) < percEvil:
            models[rcvrID] = OBU(7, epochs = subEpochs, gpu=gpu, lr = lr, motors = motors, units = units, evil = True)
        else:
            models[rcvrID] = OBU(7, epochs = subEpochs, gpu=gpu, lr = lr, motors = motors, units = units)
        # Create Slice of dataset
        vehicle = data.DataLoader(data.TensorDataset(vehicle[:,:,3:10].float(), vehicle[:,:,11].long()), batch_size=batchSize, shuffle=False, num_workers=16, persistent_workers = True) # type: ignore
        # Add sub - dataset to dataset
        rcvrs.append(rcvrID)
        dataSets[rcvrID]=vehicle
        models[rcvrID].dataset = vehicle

# Train individual models and combine
for epoch in range(epochs):
    i = 0
    if randomVehicles:
        # choose random assortment of vehicles to train on
        cars = []
        rcvrs = []
        for ted in np.random.choice(len(fedDataSet), vehicles, replace = False):
            cars.append(fedDataSet[ted])
        # Divide dataset of recieving vehicles among OBUs
        for vehicle in cars: # 10
            rcvrID = int(vehicle[0,0,2].item())
            # Add new OBU for each model
            if rcvrID not in models:
                if np.random.randint(0,100) < percEvil:
                    models[rcvrID] = OBU(7, epochs = subEpochs, gpu=gpu, lr = lr, motors = motors, units = units, evil = True)
                else:
                    models[rcvrID] = OBU(7, epochs = subEpochs, gpu=gpu, lr = lr, motors = motors, units = units)
            # Create Slice of dataset
            vehicle = data.DataLoader(data.TensorDataset(vehicle[:,:,3:10].float(), vehicle[:,:,11].long()), batch_size=batchSize, shuffle=False, num_workers=16, persistent_workers = True)
            # Add sub - dataset to dataset
            rcvrs.append(rcvrID)
            dataSets[rcvrID]=vehicle
            models[rcvrID].dataset = vehicle

    print(rcvrs, file = open('rcvrs.txt', 'w'))
    # Baseline model to add everything to. !!Do I want this or should it be a completely new model?!! Got 0% on combination before, testing with new model for next model.
    nextModel = OBU(7, epochs= subEpochs, gpu = gpu, lr = lr, motors = motors, units = units)
    # Train models
    weights = []
    for rcvr in rcvrs:
        log.startEpochTimer()
        log.startVehicleTimer()
        # Make multithreaded?
        if doValidation and models[rcvr].prevAccuracy != 0:
            _, _, p, _ = mainModel.test(inTest, outTest)
            print(f"Current Epoch: {epoch}, Reciever: {rcvr}")
            print(f"Tested. main perc: {p}, my perc: {models[rcvr].prevAccuracy}")
            if p > models[rcvr].prevAccuracy:
                print("Updating Model")
                # set model to main model, and train that 
                models[rcvr].setState(mainModel.getState())
        else:
            models[rcvr].setState(mainModel.getState())
        # Reset the trainer (should be unneeded now) to allow for further training
        # mod.resetTrainer()
        # Actually train
        models[rcvr].updateSavedStates()
        modLoss = models[rcvr].step(subEpochs)

        weights.append(1/modLoss)
        if deepTest or doValidation:
            _, _ , perc, _ = models[rcvr].test(inTest, outTest)
            percs[rcvr] = perc
            models[rcvr].prevAccuracy = perc
            # Test individual model
            print(rcvr)
            if rcvr not in results:
                results[rcvr] = ([epoch, i+1, perc, modLoss.item()])
            else:
                results[rcvr].append([epoch, i+1, perc, modLoss.item()])
        states[rcvr] = (models[rcvr].getState())
        i+=1
        log.endEpochTimer()
        log.endVehicleTimer()
    # Create combined model
    # combine models
    weights = np.abs(weights)/np.sum(weights)

    log.updateLogs([models[rcvr] for rcvr in rcvrs], epoch)

    y=0
    for w in weights:
        y+= w
    teds = [[float(weights[n]), percs[n]] for n in range(len(percs))]
    percentages.append(teds)
    histWeights.append([weights,y])
    tom = 1/i
    for rcvr in rcvrs:
        if weighing:
            nextState = nextModel.setState(nextModel.getSavedState(), dict((n, states[rcvr].get(n, 0)*weights[rcvr]) for n in states[rcvr])) # Done with weights
        else:
            nextState = nextModel.setState(nextModel.getSavedState(), states[rcvr]) # No Weights
    if weighing:
        mainModel.setState(nextState) #Weights
    else:
        mainModel.setState(dict((n, nextState.get(n, 0)/i) for n in nextState)) # No weights
# Test combined model at end
log.finalLogs([models[rcvr] for rcvr in rcvrs], percEvil)
perc = mainModel.test(testDataIn, testDataOut)
results['FINAL'] = [-1, -1, perc]
evils = []
for rcvr in rcvrs: # Create list of evil/bad vehicles
    if models[rcvr].isEvil():
        evils.append(models[rcvr].id)
print(evils, file=open(f'out/{path}VehicleStatus.txt','w'))
print(results, file = open(f'out/{path}results.txt', 'w'))
print(histWeights, file = open(f'out/{path}Weights.txt', 'w'))
print(percentages, file = open(f'out/{path}Percs.txt', 'w'))
log.log()


# 'Model got 340703/1247740 right. Accuracy: 0.27305608540240756, Precision: 0.27978235144854047, Recall: 0.919080118694362, F1 Score: 0.42897730670851897'
# This was with scheduler, 50, 10, 100. Before accuracy with this was 96.388%, now 27.306%. 
# Running test with 50, 5, 50, w/out scheduler: 'Model got 1170620/1247740 right. Accuracy: 0.9381922515908763, Precision: 0.8443495151097161, Recall: 0.9709495548961424, F1 Score: 0.90323495386345'
# Accuracy of 93.820% after half the vehicles and half the epochs. I think we need to rething the scheduler.