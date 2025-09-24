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
dataFile = 'Data/CfCMultiExtension/Disruptive_1416.csv'
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
# Create dataset of 1/100th of the entries for quicker testing during development
for index in range(leng):
    if not (int(index/10) % 300):
        tinyTestIn.append(dataSet[index, :, 3:10])
        tinyTestOut.append(dataSet[index, :, 11])
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
    
class CNNLSTM(nn.Module):
    def __init__(self, inputSize, outputDim, lSTMinFeat = 20, batchFirst = True):
        super(CNNLSTM, self).__init__()
        self.inputSize = inputSize
        self.outputDim = outputDim
        self.lSTMinFeat = lSTMinFeat
        self. batchFirst = batchFirst
        # CNN layer 
        self.cnn = nn.Conv1d(inputSize, lSTMinFeat, kernel_size=3)
        # LSTM Layers in of LSTM should be same as out of CNN - 20 x 
        self.lstm = nn.LSTM(lSTMinFeat, 256, 4, batch_first=batchFirst)
        # Linear Layer (SVM)
        self.linear = nn.Linear(lSTMinFeat, outputDim)

    def forward(self, batch, hidden = None):
        batch, hidden = self.cnn(batch, hidden)
        batch, hidden = self.lstm(batch, hidden)
        batch, hidden = self.linear(batch, hidden)
        out = nn.functional.relu(batch)
        return out, hidden

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
        # if self.learner.getHidden() != None and other.learner.getHidden() != None:
        self.learner.load_state_dict(dict( (n, self.learner.state_dict().get(n, 0)+other.learner.state_dict().get(n, 0)) for n in set(self.learner.state_dict())|set(other.learner.state_dict()) ))
        # elif other.learner.getHidden() != None:
        #     self.model.load_state_dict(other.model.state_dict())
        # elif self.learner.getHidden() != None:
        #     self.model.load_state_dict(self.model.state_dict())
        return self
    
    def __mul__(self, i):
        self.learner.load_state_dict(dict((n, self.learner.state_dict().get(n, 0)*i) for n in self.learner.state_dict()))
        return self

    # Overloading div. function to average model
    def __truediv__(self, i):
        
        self.learner.load_state_dict(dict((n, self.learner.state_dict().get(n, 0)/i) for n in self.learner.state_dict()))
        # self.model.load_state_dict(self.model.state_dict()/i)
        # self.learner.setHidden(self.learner.getHidden() / i)
        # self.model.fF.weight = nn.Parameter(self.model.fF.weight/i)
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
        # Calculate percent correct and percent zero
        if mathy:
            if Pt != 0:
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
                accuracy = (Pt+Nt)/(Pt+Pf+Nf+Nt)
                print("Model could not complete tests.")
                return 0, 0, 0, accuracy 
        else:
            if Pt != 0:
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
    
    def testStep(self, dataLoader):
        self.learner.validation_step(next(iter(dataLoader)), 0)
    
    def setModel(self, model):
        if not model == None:
            self.model = model

    def getModel(self):
        return self.model
    
    def getSavedState(self):
        return self.prevWeights
    
    def updateSavedStates(self):
        if self.evil:
            self.prevWeights = dict((n, torch.full(self.learner.state_dict()[n].shape,10000000)) for n in self.learner.state_dict()).copy()
            return # Never update weights, so always passing on Zero weights. Can also try with infite/random weights
        self.prevWeights = self.learner.state_dict().copy()
    
    def getState(self):
        return self.learner.state_dict()
    
    def restoreFromBackup(self):
        self.trainer.fit_loop.max_epochs = self.trainer.current_epoch - self.perEpoch
        self.trainer.fit(self.learner, self.dataset, ckpt_path=f'log/Fed/{self.id}checkpoint.ckpt')
        # self.model.load_state_dict(self.backupWeights['model'])
        # self.learner.load_state_dict(self.backupWeights['learner'])

    def saveBackup(self):
        self.trainer.save_checkpoint(f'log/Fed/{self.id}checkpoint.ckpt')
        # self.backupWeights['model'] = self.model.state_dict().copy()
        # self.backupWeights['learner'] = self.learner.state_dict().copy()
    
    def isEvil(self):
        return True if self.evil else False
    
    def setState(self, one, two = None):
        if two:
            tom = dict((n, one.get(n, 0)+two.get(n, 0)) for n in set(one)|set(two))
        else:
            tom = one
        self.learner.load_state_dict(tom)
        return tom
    
    def step(self, epochs):
        self.perEpoch = epochs
        self.trainer.fit_loop.max_epochs = self.trainer.current_epoch + epochs
        self.curr_loss = self.fit(self.dataset).item()
        return self.curr_loss

    def updateSelected(self):
        # self.sampling = [self.nearbyOBUs[i] for i in torch.utils.data.WeightedRandomSampler([self.samplingWeights[i] for i in self.nearbyOBUs], self.outnum)] # Randomly generates list of outnum vehicles to sample based on the sampling weights.
        # print(self.sampling)
        # return self.sampling
        self.sampling = []
        count = 0
        for idx in self.nearbyOBUs:
            rand = np.random.randint(0, 100)
            if rand <= int(100*self.samplingWeights[idx]): # Less than or equal, as we want 1 to be selected every time.
                self.sampling.append(int(idx))
                count += 1
        return self.sampling # Returning how many vehicles were selected for training
        
        
    def resetTrainer(self):
        self.trainer = pl.Trainer(
            logger = CSVLogger('log'), # Set ouput destination of logs, logging accuracy every 50 steps
            max_epochs = self.epochs, # Number of epochs to train for
            gradient_clip_val = 1, # This is said to stabilize training, but we should test if that is true
            accelerator = "gpu" if self.gpu else "cpu" # Using the GPU to run training or not
            )



# OBU module class to organize
class lstmOBU():
    def __init__(self, inputSize = 10, units = 20, motors = 8, outputs = 20, epochs = 0, lr = 0.001, gpu = False):
        self.lr = lr
        self.epochs = epochs
        self.gpu = gpu
        self.model = CNNLSTM(inputSize, outputs)
        self.learner = CfCLearner(self.model, lr) # tune units, lr
        self.trainer = pl.Trainer(
            logger = CSVLogger('log/Non-Fed'), # Set ouput destination of logs, logging accuracy every 50 steps
            max_epochs = epochs, # Number of epochs to train for
            gradient_clip_val = 1, # This is said to stabilize training, but we should test if that is true
            accelerator = "gpu" if gpu else "cpu" # Using the GPU to run training or not
            )
        self.curr_loss = None

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
    
    def step(self, epochs, dataLoader):
        self.trainer.fit_loop.max_epochs = self.trainer.current_epoch + epochs
        self.curr_loss = self.fit(dataLoader).item()
    
    def train(self, epochs, dataLoader, log):
        epoch = 0
        while epoch < epochs:
            log.startEpochTimer()
            log.startVehicleTimer()
            self.step(1, dataLoader)
            log.endEpochTimer()
            log.endVehicleTimer()
            log.updateLogs([self], epoch)
            epoch += 1
        log.finalLogs(0)
        log.log()

    
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

    def startVehicleTimer(self):
        self.startTime = time.time()
    
    def endVehicleTimer(self):
        self.times.append(time.time()-self.startTime)

    def startEpochTimer(self):
        self.startEpochTime = time.time()
    
    def endEpochTimer(self):
        self.epochTimes.append(time.time()-self.startEpochTime)

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
            

    def finalLogs(self, vehicles, percEvil):
        self.lossVPercEvil = [percEvil, self.avgLossVEpoch[-1][1]]
        self.F1VPercEvil = [percEvil, self.avgF1VEpoch[-1][1]]
        self.RecallVPercEvil = [percEvil, self.avgRecallVEpoch[-1][1]]
        self.PrecisionVPercEvil = [percEvil, self.avgPrecisionVEpoch[-1][1]]
        self.AccuracyVPercEvil = [percEvil, self.avgAccuracyVEpoch[-1][1]]
        self.AvgVehicleTime = np.sum(self.times)/len(self.times)
        self.MaxVehicleTime = np.max(self.times)
        self.TotTime = np.sum(self.epochTimes)/len(self.epochTimes)

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



            # DeFTA: Decentralized Federalized Training

pl.seed_everything(1000)

vehicleNumTot = 200 # 50 # 50
subNetworkNum = 45 # 15 # 15
totEpochs = 60 # 30 # 5
stepsPerEpoch = 10 # 5 # 30
stepsPerTestingEpoch = 15
firstSteps = 50
minConnnectedVehicles = 25 # 10
backupThreshold = 0.1
lossBackupThreshold = 3
vehicles = []
selectionWeights = {}
gpu = False
doEvil = False
percEvil = 0
lr = 0.01
phiGain = 1
phiGainLoss = 1
testingRoundNum = 10

path = f"DeFTA/Disruptive-CNN-{doEvil}-{percEvil}-{vehicleNumTot}-{subNetworkNum}-{totEpochs}-{stepsPerEpoch}-{stepsPerTestingEpoch}-{minConnnectedVehicles}-{backupThreshold}-{phiGain}/"
if not os.path.exists(f"out/{path}"):
    os.makedirs(f"out/{path}")

log = OutLogger(path)
# Function to update sampling weights and confidence values
def phi(vehicle):
    m = [0 for n in range(vehicleNumTot)]
    for t in vehicle.sampling:
        m[t] += 1  # define matrix m that contains weather the vehicle is in the sampled set, and how many times it is in the set.
    _, _, _, vehicle.curr_acc = vehicle.test(tinyTestIn, tinyTestOut, True)
    if vehicle.curr_acc != 0: # If the model doesn't just test to 0
        '''RUN TEST, use result of this test and change to use prev_acc, curr_acc, and create a backup Threshold for the accuracy.'''
        if vehicle.prev_acc != None: # If it isn't the first iteration:
            if vehicle.prev_acc-vehicle.curr_acc > backupThreshold: # If this training round broke the model
                print(f"Previous Acc: {vehicle.prev_acc}, Current Acc: {vehicle.curr_acc}")
                print("Loading From Backup")
                vehicle.restoreFromBackup() # Restore backup weights
                vehicle.step(stepsPerEpoch) # Training Step
                vehicle.trust_acc = -4 # set loss to infinity, as this model is destroyed
            else:
                if vehicle.curr_acc > vehicle.prev_acc: # If this is new best model
                    vehicle.saveBackup() # Save model as new backup
                vehicle.trust_acc = vehicle.curr_acc - vehicle.prev_acc # Set change in trust
                vehicle.prev_acc = vehicle.curr_acc # Update previous loss to stay current
            print(vehicle.confidences)
            for i in range(len(vehicle.confidences)):
                if m[i]*vehicle.otherPriorities[i]*vehicle.trust_acc > 0:
                    add = m[i]*vehicle.otherPriorities[i]*vehicle.trust_acc
                else:
                    add = phiGain*m[i]*vehicle.otherPriorities[i]*vehicle.trust_acc
                vehicle.confidences[i] = vehicle.confidences[i] + (add) # Update confidences based on what vehicles are contributing to the training and the size of their contribution
                print(f'{i}: {add}')
            for i in range(len(vehicle.confidences)):
                if vehicle.confidences[i] > 5:
                    vehicle.confidences[i] = 5 # Add max to the sampling weights
            for i in range(len(vehicle.samplingWeights)):
                vehicle.samplingWeights[i] = (0.2 * vehicle.confidences[i]) if (vehicle.confidences[i] > 0) else (vehicle.confidences[i]) # Perform cRELU on C, with weighting a as 0.2 as per the findings of the DeFTA paper
            # vehicle.samplingWeights = np.exp(vehicle.samplingWeights)/np.sum(np.exp(vehicle.samplingWeights)) # Implementation of the softMax function to align the theta values properly. Trying without this to have everything sampled unless issue.
            for i in range(len(vehicle.samplingWeights)):
                if vehicle.samplingWeights[i] > 1:
                    vehicle.samplingWeights[i] = 1 # Add max to the sampling weights
        else:
            vehicle.prev_acc = vehicle.curr_acc # Update previous loss while allowing for one epoch of randomness
        vehicle.prev_loss = vehicle.curr_loss
    else: # If the model tests to zero, revert to previous loss-based method
        if vehicle.prev_loss != None: # If it isn't the first iteration:
            if vehicle.curr_loss > vehicle.prev_loss * lossBackupThreshold: # If this training round broke the model
                print(f"Previous Loss: {vehicle.prev_loss}, Current loss: {vehicle.curr_loss}")
                print("Loading From Backup")
                vehicle.restoreFromBackup() # Restore backup weights
                vehicle.step(stepsPerEpoch) # Training Step
                vehicle.trust_loss = 10 # set loss to infinity, as this model is destroyed
            else:
                if vehicle.curr_loss < vehicle.prev_loss: # If this is new best model
                    vehicle.saveBackup() # Save model as new backup
                vehicle.trust_loss = vehicle.curr_loss - vehicle.prev_loss # Set change in trust
                vehicle.prev_loss = vehicle.curr_loss # Update previous loss to stay current
            print(vehicle.confidences)
            for i in range(len(vehicle.confidences)):
                if m[i]*vehicle.otherPriorities[i]*vehicle.trust_loss < 0:
                    add = m[i]*vehicle.otherPriorities[i]*vehicle.trust_loss
                else:
                    add = phiGainLoss*m[i]*vehicle.otherPriorities[i]*vehicle.trust_loss
                vehicle.confidences[i] = vehicle.confidences[i] - (add) # Update confidences based on what vehicles are contributing to the training and the size of their contribution
                print(f'{i}: {add}')
            for i in range(len(vehicle.confidences)):
                if vehicle.confidences[i] > 5:
                    vehicle.confidences[i] = 5 # Add max to the sampling weights
            for i in range(len(vehicle.samplingWeights)):
                vehicle.samplingWeights[i] = (0.2 * vehicle.confidences[i]) if (vehicle.confidences[i] > 0) else (vehicle.confidences[i]) # Perform cRELU on C, with weighting a as 0.2 as per the findings of the DeFTA paper
            # vehicle.samplingWeights = np.exp(vehicle.samplingWeights)/np.sum(np.exp(vehicle.samplingWeights)) # Implementation of the softMax function to align the theta values properly. Trying without this to have everything sampled unless issue.
            for i in range(len(vehicle.samplingWeights)):
                if vehicle.samplingWeights[i] > 1:
                    vehicle.samplingWeights[i] = 1 # Add max to the sampling weights
        else:
            vehicle.prev_loss = vehicle.curr_loss # Update previous loss while allowing for one epoch of randomness.
    if vehicle.id in selectionWeights:
        selectionWeights[vehicle.id].append([vehicle.samplingWeights[:]])
    else:
        selectionWeights[vehicle.id] = [[vehicle.samplingWeights[:]]]
    # Returns nothing, since all operations are done inside the vehicle

def noLuckPhi(vehicle):
    m = [0 for n in range(vehicleNumTot)]
    for t in vehicle.sampling:
        m[t] += 1  # define matrix m that contains whether the vehicle is in the sampled set, and how many times it is in the set.
    vehicle.curr_f1, _, _, vehicle.curr_acc = vehicle.test(tinyTestIn, tinyTestOut, True)
    if vehicle.testing:
        print("TESTING ROUND")
        vehicle.rounds = 0
        if vehicle.prev_f1:
            if vehicle.curr_f1 >= vehicle.prev_f1-0.05: # If vehicle helps our model
                for i in range(vehicleNumTot):
                    if m[i] != 0:
                        vehicle.goodNeighbors.append(i)
                print(vehicle.goodNeighbors)
            # Keep prev_accuracy and backup constant during testing phase
            vehicle.restoreFromBackup() # Restore backup weights in order to keep best model during testing 
            print("Restored")
        else:
            vehicle.prev_acc = vehicle.curr_acc
            vehicle.prev_f1 = vehicle.curr_f1
            print('Saved')
            vehicle.saveBackup() # Save Backup for when we run training.


        if len(vehicle.toTest):
            vehicle.sampling = [vehicle.toTest.pop()]
        else:
            vehicle.testing = False
            vehicle.sampling = vehicle.goodNeighbors.copy()
    else:
        vehicle.rounds += 1
        print("PREDICTING ROUND")
        if vehicle.prev_f1: # If we have previous data to go off of
            if vehicle.curr_f1 >= vehicle.prev_f1: # If model gets better or stays the same
                vehicle.saveBackup() # Save model
                vehicle.prev_f1 = vehicle.curr_f1 # update acc
                vehicle.prev_acc = vehicle.curr_acc
            elif vehicle.curr_f1 > vehicle.prev_f1 - 0.1: # Update model but do not update saved backup model
                vehicle.prev_f1 = vehicle.curr_f1
            else: # If model is ruined
                print(f"Previous f1: {vehicle.prev_f1}, Current f1: {vehicle.curr_f1}")
                print("Loading From Backup")
                vehicle.restoreFromBackup() # Restore backup weights
                vehicle.step(stepsPerEpoch) # Training Step
                vehicle.prev_f1 = vehicle.curr_f1 # update acc
                vehicle.prev_acc = vehicle.curr_acc
                # Run tests early as we have a bad-actor
                vehicle.toTest = [i for i in vehicle.nearbyOBUs] # Only test those that contributed to this model
                vehicle.sampling = [vehicle.toTest.pop()] # Initialize first sample
                vehicle.testing = True # Start testing
                vehicle.goodNeighbors = [] # Reset known good
        else:
            vehicle.prev_acc = vehicle.curr_acc
            vehicle.prev_f1 = vehicle.curr_f1
            vehicle.saveBackup() # Save Backup for when we run training.`

                    



        if vehicle.rounds > 60:
            vehicle.toTest = [i for i in vehicle.nearbyOBUs] # Test all nearby OBUs
            vehicle.sampling = [vehicle.toTest.pop()] # Sample the first OBU
            vehicle.testing = True # Start test
            vehicle.goodNeighbors = [] # reset known good
        else:
            vehicle.sampling = vehicle.goodNeighbors # Sample all known good
        
    if vehicle.id in selections:
        selections[vehicle.id].append([vehicle.sampling[:]])
    else:
        selections[vehicle.id] = [[vehicle.sampling[:]]]



def updatePriorities(vehicles):
    for i in range(vehicleNumTot):
        vehicles[i].outnum = len(vehicles[i].sampling) + 1# Update useful outnumber of each vehicle in the simulation by having outnum = num sampled vehicles

    for i in range(vehicleNumTot): # Loop through vehicles and add priority of vehicle, done in separete loop as it requires info from other vehicles
        sum = vehicles[i].datalen/vehicles[i].outnum # Start out by including this vehicle's priority
        dum = vehicles[i].datalen/vehicles[i].outnum
        if len(vehicles[i].sampling) != 0:
            for j in vehicles[i].sampling:
                sum += vehicles[j].datalen/vehicles[j].outnum
            vehicles[i].priority = (vehicles[i].datalen/vehicles[i].outnum)/(sum)
        else:
            vehicles[i].priority = 1
        if len(vehicles[i].sampling) != 0:
            for j in vehicles[i].sampling:
                vehicles[i].otherPriorities[j] = (vehicles[j].datalen/vehicles[j].outnum)/sum # fill out standard list of other priorities
                dum += vehicles[j].datalen/vehicles[j].outnum
            print(f"vehicle {i} priority: {vehicles[i].priority}, total priority of subgroup: {dum/sum}")
        else:
            print(f'Vehicle {i} not sampling any vehicles this cycle.')

for i in range(vehicleNumTot):
    set = data.DataLoader(data.TensorDataset(fedDataSet[i][:,:,3:10].float(), fedDataSet[i][:,:,11].long()), batch_size=batchSize, shuffle=False, num_workers=10, persistent_workers = True) # Create datasets
    if doEvil:
        if np.random.randint(0,100) < percEvil:
            vehicles.append(lstmOBU(inputSize=7, units=20, motors=8, outputs=20, lr=lr, randInt=i, gpu=gpu, dataset=set, evil=True)) # Create evil vehicles
        else:
            vehicles.append(lstmOBU(inputSize=7, units=20, motors=8, outputs=20, lr=lr, randInt=i, gpu=gpu, dataset=set)) # Create vehicles
    else:
        vehicles.append(lstmOBU(inputSize=7, units=20, motors=8, outputs=20, lr=lr, randInt=i, gpu=gpu, dataset=set)) # Create vehicles
    vehicles[i].prevWeights = vehicles[i].getState() # Save previous state, so that we can do it in iterations
    vehicles[i].datalen = fedDataSet[i].shape[0]
    vehicles[i].outnum = np.random.randint(minConnnectedVehicles, subNetworkNum) # Get number of vehicles in sub network, at least #x so that vehicle has some use.
    range1 = np.arange(0, i).tolist()
    range2 = np.arange(i+1, vehicleNumTot).tolist()
    vehicles[i].nearbyOBUs = np.random.choice(range1 + range2, vehicles[i].outnum, replace=False) # Create subnetworks and add them to the vehicle
    vehicles[i].confidences = np.full((vehicleNumTot), 3.) # Initialize confidence values for all vehicles - can be shifted to a dict later to allow for varying number/discovery of vehicles
    vehicles[i].samplingWeights = np.full((vehicleNumTot), .5) # Initialize the sampling weights to 0.5 - similarly, can be switched to a dict
    vehicles[i].otherPriorities = np.zeros((vehicleNumTot)) # Initialize list of priorities
    vehicles[i].sampling = [] # Start by sampling no vehicles
    vehicles[i].id = i # Save vehicles id
    vehicles[i].toTest = [t for t in vehicles[i].nearbyOBUs]
    vehicles[i].testing = True
    vehicles[i].goodNeighbors = []

historicLoss = {}
selections = {}
results = {}

for epoch in range(totEpochs):
    print(f"Starting Epoch {epoch} now")
    updatePriorities(vehicles)
    for vehicle in vehicles:
        vehicle.updateSavedStates() # Save current model as model to send to others, so that they are getting the latest after each loop
    for vehicle in vehicles:
        log.startEpochTimer()
        log.startVehicleTimer()
         # Model aggregation - Sum weights of all participating models weighted by their priority
        sum = None
        for i in vehicle.sampling:
            if not sum:
                state = vehicles[i].getSavedState()
                sum = dict((n, state.get(n, 0)*vehicle.otherPriorities[i]) for n in state) # Multiplying weights and priority
            else:
                state = vehicles[i].getSavedState()
                add = dict((n, state.get(n, 0)*vehicle.otherPriorities[i]) for n in state) # Multiply weights and priority
                sum = dict( (n, add.get(n, 0)+sum.get(n, 0)) for n in sum) # Add this models weights to the sum
        state = vehicle.getSavedState()
        add = dict((n, state.get(n, 0)*vehicle.priority) for n in state)
        if sum:
            sum = dict((n, add.get(n, 0)+sum.get(n, 0)) for n in sum) # Add this vehicles model to the aggregation
        else:
            sum = add
        vehicle.setState(sum)
        if epoch == 0:
            loss = vehicle.step(firstSteps)
        elif vehicle.testing:
            loss = vehicle.step(stepsPerTestingEpoch) # Run training step to progress model
        else:
            loss = vehicle.step(stepsPerEpoch) # Run training step to progress model
        if vehicle.id in historicLoss:
            historicLoss[vehicle.id].append(loss)
        else:
            historicLoss[vehicle.id] = [loss]
        noLuckPhi(vehicle) # Update theta and confidence matrix
        log.endEpochTimer()
        log.endVehicleTimer()
    log.updateLogs(vehicles, epoch)

sampleSizes = {}
for id in selections:
    for epoch in selections[id]:
        if id in sampleSizes:
            sampleSizes[id].append(len(epoch))
        else:
            sampleSizes[id] = [len(epoch)] # Printing number of vehicles sampled by each vehicle each iteration
print(sampleSizes, file=open(f'out/{path}SampleSizes.txt', 'w'))
print(historicLoss, file=open(f'out/{path}HistoricLoss.txt', 'w'))
print(selections, file=open(f'out/{path}SelectedVehicles.txt', 'w'))
avgF1 = 0
evils = []
log.finalLogs(vehicles, percEvil)
for vehicle in vehicles: # Create list of evil/bad vehicles
    if vehicle.isEvil():
        evils.append(vehicle.id)
print(evils, file=open(f'out/{path}VehicleStatus.txt','w'))
print(selectionWeights, file = open(f'out/{path}SelectionWeights.txt', 'w')) # Print out the chances of selecting each vehicle
log.log()

'''
Results:
- 0 Starting weight: Accuracy: 0.2970971516501835, Precision: 0.2970971516501835, Recall: 1.0, F1 Score: 0.4580954499394479
-.5 Starting weight: Accuracy: 0.2970971516501835, Precision: 0.2970971516501835, Recall: 1.0, F1 Score: 0.4580954499394479
'''

'''
Pseudocode:
nodes = list of nodes in simulation
    every node has: N = list of nodes in network; C = list of confidence values; theta = list of weights for each vehicles contribution; D = local dataset; d = outdegree; w = local weights for model
N = list of networks (each node i has a list of attached nodes) * At some point make this vary per iteration?
for each node in simulation in parrallel:
    Initialize list of nodes to be sampled - S
        *** How do we do this ***
    for each epoch:
        collect weights from each node in N
        new w = sum(sample weight * model weight) for model in sampled models - aggregate w
        w = w - (learning rate) * (Loss function (model)) - Local optimizing (training step)
        C_i, theta_i = phi(c, model) - update c and theta
        Update S? - WeightedSample(W, theta)
        Send local data to other models - maybe not necessary depending on how I do this
        wait for the rest of peers to finish this loop

        c is based on the training loss, ie. c_i_j increases if the training loss of w_i decreases (from their paper, they suggest coming up with your own)
        theta = softmax(cRELU(c))
        cRELU = {x; x<= 0, ax; x >0} They suggest a = 0.2 from their findings

        phi(c, w_i):
            m is binary matrix where its j-th entry is 1 iff j is in the set of used models (S) - m has all values on different columns
            p is the set of aggregation weights for all nodes in the network - P has all values on different rows
            if w_i is bad:
                w = w_backup - don't allow bad training to mess up model
                do training step on w
                loss_trust = inf.
            else:
                if loss_curr is lowest:
                    w_backup = w - create backup
                loss_trust = loss_curr - loss_last
                loss_last = loss_curr
            c = c_prev - (m composite multipied by p) * loss_trust - update c
            theta = softmax(cRELU(c)) - Update sample weights
            Update S with subset of N, random sampling using weights to randomize selection size as well as selected values


'''