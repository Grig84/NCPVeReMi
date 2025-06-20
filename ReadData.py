#GENERATE DATA FILES:
import json
import pandas as pd
import os

#Reading data as described in Gong et al.
#Time sequences are 10 timepoints (Messages) with 7 features per message.
#Organized by car.

attackFolder = "/Users/will/LocalDocuments/Kettering/VeReMi-Dataset-Expanded/" #Folder where all of the logs are
path = "DoS_1416/VeReMi_54000_57600_2022-9-11_19:12:56/" #Current subfolder to add together
fileName = path.split("/")[0] #Extracting attack type
outputFolder = "/Users/will/LocalDocuments/Kettering/NCPVeReMi/Data/" #Where csvs will go
vehicles = [] #Stores an array with vehicle ID and attacker status
dataset = {} #Stores output while program is running

for log in os.listdir(attackFolder+path):
    if log.endswith('.json') and not log.startswith("traceGr"): #Get all files without ground truth
        name = log.split("-")
        vID, attk = name[2], int(name[3][1:]) #Get ID and attacker status, casting attacker status to int
        vehicles.append([vID,attk]) #Append ID and attacker status to running list
        data = pd.read_json(attackFolder+path+log, lines=True, ) #Get all of the entries
        data = data.loc[data['type'] == 2] #Only gather entries from this vehicle (Do we want this? or do we want the opposite??????)
        data = data.dropna(axis=1, how="all") #Remove null columns
        data = data.drop(axis=1, labels='type') #Remove useless column
        data = data.reset_index(drop=True) #reorder the table to have consecutive indexes
        list = data.values.tolist() #Convert dataframe to list for ease of reformatting (probably not necessary, and wasteful)
        dic = data.to_dict(orient="index") #Convert dataframe to dictionary to preserve formatting and labels
        dataset[vID] = {"BSMs":dic, "Attack Type":attk} #Organize data as a dict for ease of reference

json.dump(dataset, open(outputFolder+fileName+".json", "w"), indent=4) #Output data as nice looking json file.