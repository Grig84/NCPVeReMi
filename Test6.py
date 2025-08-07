#Imports:
from ncps.wirings import AutoNCP
from ncps.torch import CfC
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from numpy import genfromtxt
import numpy as np
import torch
import torch.utils.data as data
import matplotlib as plt
import torch.nn as nn
import os
import time
import json
import csv

torch.set_float32_matmul_precision("high")

pl.seed_everything(1000)

#Dataset Formatting
#Generate Time sequences that are 10 timepoints (Messages) with 7 features per message.
#Organized by car.

#Current Simulation File
dataFile = 'Data/CfCMultiExtension/ConstPos_0709.csv' # DoS_0709

dataSet = genfromtxt(dataFile, delimiter=',')
batchSize = 64
# Ceate dataloader and fill with (BSM, attk#). Expanding to add 0th dimension for batches.
# Batch size should be 64 for the low density simulations and 128 for high density simulations.
# No shuffle to keep batches on same vehicle.
# Num_workers is set to = num CPU cores
dataSet[0:-1,:] = dataSet[1:,:] # Get rid of the first null value of the dataset
print(dataSet.shape)
# count subsets per vehicle
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
# Create new arrays per vehicle for federated learning
splits = np.split(dataSet, np.cumsum(counts)[:-1])
# Create seperate datasets for testing and training, using Train Percentage as metric for split
trainDataIn = torch.Tensor(dataSet[:int(leng*(trainPerc/100)),:,3:10]).float() # 1
trainDataOut = torch.Tensor(np.int_(dataSet[:int(leng*(trainPerc/100)),:,11])).long()
testDataIn = torch.Tensor(dataSet[int(leng*(trainPerc/100)):,:,3:10]).float() # 1
testDataOut = torch.Tensor(np.int_(dataSet[int(leng*(trainPerc/100)):,:,11])).long()
newsetIn = []
newsetOut = []
testsetIn = []
testsetOut = []
# Create dataset of 1/100th of the entries for quicker testing during development
for index in range(0,int(leng * (trainPerc/100))):
    if not (int(index/10) % 10):
        newsetIn.append(dataSet[index,:,3:10]) # 1
        newsetOut.append((dataSet[index,:,11]))
testingIn = torch.Tensor(np.array(newsetIn)).float()
testingOut = torch.Tensor(np.array(newsetOut)).long()
for idx in range(int((leng) * (trainPerc/100)), leng):
    if not (int(idx/10) % 10):
        testsetIn.append(dataSet[idx,:,3:10])
        testsetOut.append((dataSet[idx,:,11]))
testingIn = torch.Tensor(np.array(newsetIn)).float()
testingOut = torch.Tensor(np.array(newsetOut)).long()
inTest = torch.Tensor(np.array(testsetIn)).float()
outTest = torch.Tensor(np.array(testsetOut)).long()
# Create Dataloaders for all the datasets
dataLoaderTrain = data.DataLoader(data.TensorDataset(trainDataIn, trainDataOut), batch_size=batchSize, shuffle=False, num_workers=10, persistent_workers = True, drop_last= True)
dataLoaderTest = data.DataLoader(data.TensorDataset(testDataIn, testDataOut), batch_size=batchSize, shuffle=False, num_workers=10, persistent_workers = True, drop_last= True)
testingDataLoader = data.DataLoader(data.TensorDataset(testingIn, testingOut), batch_size=batchSize, shuffle = False, num_workers=10, persistent_workers = True, drop_last= True)
testingTestData = data.DataLoader(data.TensorDataset(testingIn, testingOut), batch_size=batchSize, shuffle = False, num_workers=10, persistent_workers = True, drop_last= True)
print(dataSet.shape)

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
            f1, recall, precision, accuracy = vehicle.test(inTest, outTest, True)
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
            

    def finalLogs(self, percEvil):
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
        # self.lr
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = 0.001)
        return ([optimizer], [torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.6)])
    

class Modena(nn.Module): 
    # CfC with feed-forward layer to classify at end.
    def __init__(self, inputSize, unitNum, motorNum, outputDim, batchFirst = True):
        super().__init__()
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
    

# OBU module class to organize
class OBU():
    def __init__(self, inputSize = 9, units = 20, motors = 8, outputs = 20, epochs = 0, lr = 0.001, gpu = False):
        self.lr = lr
        self.epochs = epochs
        self.gpu = gpu
        self.model = Modena(inputSize, units, motors, outputs)
        self.learner = CfCLearner(self.model, lr) # tune units, lr
        self.trainer = pl.Trainer(
            logger = CSVLogger('log/Non-Fed'), # Set ouput destination of logs, logging accuracy every 50 steps
            max_epochs = epochs, # Number of epochs to train for
            gradient_clip_val = 1, # This is said to stabilize training, but we should test if that is true
            accelerator = "gpu" if gpu else "cpu" # Using the GPU to run training or not
            )
        self.curr_loss = None
    
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
                print("Model could not complete tests.")
                return 0, 0, 0, 0
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


epochs = 30
lr = 0.01
testOBU = OBU(
    inputSize = 7, # 9  # Number of features per BSM
    units = 20, # Number of hidden cells
    motors = 8, # Number of motor neurons
    outputs = 20, # Number of possible labels
    lr = lr, # 0.001
    gpu = False
)
path = f"Normal/ConstPos-{epochs}-{lr}/"

log = OutLogger(path)