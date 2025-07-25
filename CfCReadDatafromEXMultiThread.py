#GENERATE DATA FILES FOR CFC MODEL FROM VEREMI:
from datetime import timedelta
import time
import pandas as pd
import os
import multiprocessing

#Large difference in reading comes from accessing files, not reading.
#NO ACCELERATION DATA
#Reading data as described in Gong et al.
#Time sequences are 10 timepoints (Messages) with 7 features per message.
#Organized by car.

#Windows:
attackFolder = "C:/Users/will/Documents/school/Kettering/VeX/" #Folder where all of the logs are
fileName = "temp" #Extracting attack type
outputFolder = "C:/Users/will/Documents/school/Kettering/NCPVeReMi/Data/CfCMultiExtension/" #Where csvs will go

#Mac:
'''
attackFolder = "/Users/will/LocalDocuments/Kettering/VeReMi-Dataset-Expanded/" #Folder where all of the logs are
path = "DoS_1416/VeReMi_54000_57600_2022-9-11_19:12:56/" #Current subfolder to add together
fileName = path.split("/")[0] #Extracting attack type
outputFolder = "/Users/will/LocalDocuments/Kettering/NCPVeReMi/Data/MultiVeReMi/" #Where csvs will go
'''

def crunch(folder, threadNum):
    totLoop = 0
    fileName = folder #Extracting attack type
    vehicles = {} #Stores an array with vehicle ID and attacker status
    dataset = pd.DataFrame(columns=['RecieverID','SenderID', 'rcvTime', 'RelX','RelY','MssgCount','dVx', 'dVy', 'AttkType']) #Stores output while program is running
    #Loop through all different simulations
    for log in os.listdir(attackFolder+folder+'/'):
        if log.endswith('.json') and not log.startswith("traceGr") and not log.startswith('._'): #Get all files without ground truth
            name = log.split("-")
            vID, attk = name[1], int(name[3][1:]) #Get ID and attacker status, casting attacker status to int
            vehicles[vID] = attk #Append ID and attacker status to running list
            totLoop += 1

    #Loop through entries and save messages in correct formatting
    mssgs= {}
    print("Thread " + str(threadNum) + " Starting")
    dataset = pd.DataFrame(columns=['RecieverID','SenderID', 'rcvTime', 'RelX','RelY','MssgCount','dVx', 'dVy', 'dAx', 'dAy', 'AttkType']) #Stores output while program is running
    loop = 0
    entrys = []
    startTime = time.monotonic()
    for log in os.listdir(attackFolder+folder+'/'): #Loop through all logs
        tag = "-th"+str(threadNum)
        #if file is a log...
        if log.endswith('.json') and not log.startswith("traceGr") and not log.startswith('._'): #Get all files without ground truth
            if not loop % int(totLoop/10):
                print("Thread " + str(threadNum) + ": " + str(int(loop*100/totLoop)) + "%")
            loop += 1
            #Setup IDs
            rcvID = log.split("-")[1]
            rcvPos = [0,0]
            countsSinceLocal = {}
            prevTelems = {}
            data = pd.read_json(attackFolder+folder+"/"+log, lines=True) #Get all of the entries
            lndata = data.to_dict(orient='records')
            for row in lndata:
                if row['type'] == 2:
                    #Save Rec ID and messages since last transmission
                    rcvPos = [row['pos'][0], row['pos'][1]]
                    countsSinceLocal = {}
                else:
                    sender = row['sender']
                    #Create entry and save it to dataset
                    if sender in countsSinceLocal.keys(): #Keep track of messages since last transmission
                        countsSinceLocal[sender] += 1
                    else:
                        countsSinceLocal[sender] = 1
                    #Parse variables/features for entry:
                    if sender in prevTelems.keys(): #Get previous entry or set differentces to 0
                        prevTel = prevTelems[sender] #Get previous telemetry
                        dVx = abs((row['pos'][0] - prevTel['pos'][0]) - (row['spd'][0] - prevTel['spd'][0])/2) #get the change in spd and accel
                        dVy = abs((row['pos'][1] - prevTel['pos'][1]) - (row['spd'][1] - prevTel['spd'][1])/2)
                        dAx = abs((row['spd'][0] - prevTel['spd'][0]) - (row['acl'][0] - prevTel['acl'][0])/2)
                        dAy = abs((row['spd'][1] - prevTel['spd'][1]) - (row['acl'][1] - prevTel['acl'][1])/2)
                    else:
                        dVx = dVy = dAx = dAy = 0 #If this is the first message, no change in spd or accel
                    count = countsSinceLocal[sender]
                    relX = abs(row['pos'][0] - rcvPos[0]) #Get relative position of the sender
                    relY = abs(row['pos'][1] - rcvPos[1])
                    entry = [rcvID, int(sender), row['rcvTime'], relX, relY, count, dVx, dVy, dAx, dAy, vehicles[str(int(sender))]]#Create entry for this message
                    prevTelems[sender] = row #Update previous info with this row
                    entrys.append(entry)

                    #keep track of message count per sender
                    if sender in mssgs.keys(): #if an entry already exists:
                        mssgs[sender] = [entry, mssgs[sender][1]+1] # incriment mssg count and add latest entry
                    else:
                        mssgs[sender] = [entry, 1] # Set count to 1 and add latest entry

    # Padding:
    # go through sending vehicles
    for tom in mssgs.values():
        entry, it = tom
        while it % 100: # While number of entries per vehicle is not divisible by 100
            entrys.append(entry) # Expand messages from this sender
            it += 1 # Add one to the count

    dataset = pd.DataFrame(entrys, columns=dataset.columns) # Convert list of entries to dataframe
    #Organize dataframe:
    dataset.sort_values(by=['SenderID', "rcvTime",'RecieverID'], inplace=True) #Sort by Sender ID then by Recieve time
    dataset.reset_index(drop=True, inplace=True) #make indexes make sense after sort



    dataset.to_csv(open(outputFolder+fileName+'.csv', 'w')) #Output to CSV'
    print(fileName + " Done in " + str(timedelta(seconds=(time.monotonic()-startTime)))+ " by Thread " + str(threadNum)) #Log

fullStartTime = time.monotonic()

# crunch('DoS_0709', 1) #TEST

#Start multithreaded workload

if __name__ == '__main__':
    processes = []
    threadNum = 0
    for subfolder in os.listdir(attackFolder):
        threadNum+=1
        processes.append(multiprocessing.Process(target=crunch, args=(subfolder, threadNum)))

    for thread in processes:
        thread.start()


    for thread in processes:
        thread.join()

print("Program execution time: " + str(timedelta(seconds=(time.monotonic() - fullStartTime))))
