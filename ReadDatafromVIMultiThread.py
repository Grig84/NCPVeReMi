#GENERATE DATA FILES FOR CFC MODEL FROM VEREMI:
import json
import pandas as pd
import os
import multiprocessing

#Large difference in reading comes from accessing files, not reading.
#NO ACCELERATION DATA
#Reading data as described in Gong et al.
#Time sequences are 10 timepoints (Messages) with 7 features per message.
#Organized by car.

#Windows:
attackFolder = "C:/Users/will/Documents/school/Kettering/VeremiData - Copy/" #Folder where all of the logs are
fileName = "temp" #Extracting attack type
outputFolder = "C:/Users/will/Documents/school/Kettering/NCPVeReMi/Data/MultiVeReMi/" #Where csvs will go

#Mac:
'''
attackFolder = "/Users/will/LocalDocuments/Kettering/VeReMi-Dataset-Expanded/" #Folder where all of the logs are
path = "DoS_1416/VeReMi_54000_57600_2022-9-11_19:12:56/" #Current subfolder to add together
fileName = path.split("/")[0] #Extracting attack type
outputFolder = "/Users/will/LocalDocuments/Kettering/NCPVeReMi/Data/MultiVeReMi/" #Where csvs will go
'''

def crunch(folder, threadNum):

    fileName = "temp" #Extracting attack type
    vehicles = {} #Stores an array with vehicle ID and attacker status
    dataset = pd.DataFrame(columns=['RecieverID','SenderID', 'rcvTime', 'RelX','RelY','MssgCount','dVx', 'dVy', 'AttkType']) #Stores output while program is running
    for subfolder in os.listdir(attackFolder+folder):
        foldName = subfolder.split('.')[2]
        vehicles[foldName] = {}
        for log in os.listdir(attackFolder+folder+subfolder+"/veins-maat/simulations/securecomm2018/results/"):
            if log.endswith('.json') and not log.startswith("Gr") and not log.startswith('._'): #Get all files without ground truth
                name = log.split("-")
                vID, attk = name[2], int(name[3].split('.')[0][1:]) #Get ID and attacker status, casting attacker status to int
                vehicles[foldName][vID] = attk #Append ID and attacker status to running list

    #Loop through entries and save messages in correct formatting
    iter = 15
    for subfolder in os.listdir(attackFolder+folder):
        foldName = subfolder.split('.')[2]
        print("Thread " + str(threadNum) + " Starting")
        dataset = pd.DataFrame(columns=['RecieverID','SenderID', 'rcvTime', 'RelX','RelY','MssgCount','dVx', 'dVy', 'AttkType']) #Stores output while program is running
        for log in os.listdir(attackFolder+folder+subfolder+"/veins-maat/simulations/securecomm2018/results/"):
            if log.endswith('.sca') and not log.startswith('._'):
                if fileName == log.split('.')[0]:
                    iter += 1
                else:
                    iter = 15
                fileName = log.split('.')[0]
                tag = "th-"+str(threadNum)+"ix-"+str(iter)
            if log.endswith('.json') and not log.startswith("Gr") and not log.startswith('._'): #Get all files without ground truth
                rcvID = log.split("-")[2]
                rcvPos = [0,0]
                rcvVel = [0,0]
                #rcvAcl = [0,0]
                countsSinceLocal = {}
                prevTelems = {}
                data = pd.read_json(attackFolder+folder+subfolder+"/veins-maat/simulations/securecomm2018/results/"+log, lines=True) #Get all of the entries
                for _, row in data.iterrows():
                    if row['type'] == 2:
                        #Save Rec ID and messages since last transmission
                        rcvPos = [row['pos'][0], row['pos'][1]]
                        countsSinceLocal = {}
                    else:
                        #Create entry and save it to dataset
                        if row['sender'] in countsSinceLocal.keys(): #Keep track of messages since last transmission
                            countsSinceLocal[row['sender']] += 1
                        else:
                            countsSinceLocal[row['sender']] = 1
                        #Parse variables/features for entry:
                        if row['sender'] in prevTelems.keys(): #Get previous entry or set differentces to 0
                            prevTel = prevTelems[row['sender']] #Get previous telemetry
                            dVx = abs((row['pos'][0] - prevTel['pos'][0]) - (row['spd'][0] - prevTel['spd'][0])/2) #get the change in spd and accel
                            dVy = abs((row['pos'][1] - prevTel['pos'][1]) - (row['spd'][1] - prevTel['spd'][1])/2)
                            #dAx = abs((row['spd'][0] - prevTel['spd'][0]) - (row['acl'][0] - prevTel['acl'][0])/2)
                            #dAy = abs((row['spd'][1] - prevTel['spd'][1]) - (row['acl'][1] - prevTel['acl'][1])/2)
                        else:
                            dVx = dVy = 0 #If this is the first message, no change in spd or accel
                        count = countsSinceLocal[row['sender']]
                        relX = abs(row['pos'][0] - rcvPos[0]) #Get relative position of the sender
                        relY = abs(row['pos'][1] - rcvPos[1])
                        entry = [rcvID, int(row['sender']), row['rcvTime'], relX, relY, count, dVx, dVy, vehicles[foldName][str(int(row['sender']))]]#Create entry for this message
                        prevTelems[row['sender']] = row #Update previous info with this row
                        dataset.loc[-1] = entry #add entry to dataframe
                        dataset.index += 1

        #Organize dataframe:
        dataset.sort_values(by=['RecieverID','SenderID', "rcvTime"], inplace=True) #Sort by reciever ID then by Sender ID
        dataset.reset_index(drop=True, inplace=True) #make indexes make sense after sort

        dataset.to_csv(open(outputFolder+fileName+tag+".csv", 'w')) #Output to CSV'
        print(fileName+tag + " Done")

if __name__ == '__main__':
    processes = []
    threadNum = 0
    for subfolder in os.listdir(attackFolder):
        threadNum+=1
        processes.append(multiprocessing.Process(target=crunch, args=(subfolder+"/", threadNum)))

    for thread in processes:
        thread.start()


    for thread in processes:
        thread.join()