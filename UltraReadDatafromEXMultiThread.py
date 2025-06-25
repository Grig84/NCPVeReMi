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
outputFolder = "C:/Users/will/Documents/school/Kettering/NCPVeReMi/Data/UltraMultiExtension/" #Where csvs will go

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
    dataset = pd.DataFrame(columns=['SenderID', 'sendTime', 'Posx','Posy','Spdx','Spdy']) #Stores output while program is running
    #Loop through all different simulations
    for log in os.listdir(attackFolder+folder):
        if log.endswith('.json') and not log.startswith("traceGr") and not log.startswith('._'): #Get all files without ground truth
            name = log.split("-")
            vID, attk = name[2], int(name[3][1:]) #Get ID and attacker status, casting attacker status to int
            vehicles[vID] = attk #Append ID and attacker status to running list
            totLoop += 1

    #Loop through entries and save messages in correct formatting
    print("Thread " + str(threadNum) + " Starting")
    dataset = pd.DataFrame(columns=['SenderID', 'sendTime', 'Posx','Posy','Spdx','Spdy']) #Stores output while program is running
    loop = 0
    startTime = time.monotonic()
    for log in os.listdir(attackFolder+folder): #Loop through all logs
        tag = "-th"+str(threadNum)

        #if file is a log...
        if log.endswith('.json') and not log.startswith("traceGr") and not log.startswith('._'): #Get all files without ground truth
            if not loop % (totLoop/10):
                print("Thread " + str(threadNum) + ": " + str(int(loop*100/totLoop)) + "%")
            #Setup IDs
            data = pd.read_json(attackFolder+folder+"/"+log, lines=True) #Get all of the entries
            for _, row in data.iterrows():
                entry = [int(row['sender']), row['sendTime'], row['pos'][0], row['pos'][1], row['spd'][0], row['spd'][1]]#Create entry for this message
                dataset.loc[-1] = entry #add entry to dataframe
                dataset.index += 1

    #Organize dataframe:
    dataset.sort_values(by=['SenderID', "sendTime"], inplace=True) #Sort by reciever ID then by Sender ID
    dataset.reset_index(drop=True, inplace=True) #make indexes make sense after sort

    dataset.to_csv(open(outputFolder+fileName+tag+".csv", 'w')) #Output to CSV'
    print(fileName+tag + " Done in " + str(timedelta(seconds=(time.monotonic()-startTime))))

fullStartTime = time.monotonic()

crunch(attackFolder + "DoS_1416", 1)
#Start multithreaded workload
# if __name__ == '__main__':
#     processes = []
#     threadNum = 0
#     for subfolder in os.listdir(attackFolder):
#         threadNum+=1
#         processes.append(multiprocessing.Process(target=crunch, args=(subfolder+"/", threadNum)))

#     for thread in processes:
#         thread.start()


#     for thread in processes:
#         thread.join()

print("Program execution time: " + str(timedelta(seconds=(time.monotonic() - fullStartTime))))