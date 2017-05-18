
# coding: utf-8

# In[5]:

import csv
import matplotlib.pyplot as plt
import json
from collections import defaultdict

#Maps data in the CSV file, eg [type ID, first column axis, second column axis, third column axis]
class CsvDataMap(object):
    acc = [3, 'x', 'y', 'z']
    gyro = [4, 'x', 'y', 'z']
    mag = [5, None, None, None]
    def __init__(self, acc, gyro, mag):
        self.acc = acc
        self.gyro = gyro
        self.mag = mag

#Creates a map object based on experimentally determined type IDs and column axes
datamap = CsvDataMap([3, 'x', 'y', 'z'], [4, 'x', 'z', 'y'], [5, None, None, None])

#X,Y,Z -> 0,1,2
#nick, will, chris, luke -> 0,1,2,3

acceleration = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
gyroscope = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
velocity = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
position = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
sampling = [0,0,0,0]
samplingSec = 0
samplingCount = 0
samplingCulm = []

def setSample(i):
    global samplingCulm
    global samplingSec
    global samplingCount
    global sampling
    sampling[i] = sum(samplingCulm)/len(samplingCulm)
    samplingSec = 0
    samplingCount = 0
    samplingCulm = []

with open('NickEdit.csv') as file:
    fileobj = csv.reader(file, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for row in fileobj:
        
        if(samplingSec == 0):
            samplingSec = int(row[0])
        
        if(int(row[0]) != samplingSec):
            samplingCulm.append(samplingCount)
            samplingCount = 0
            samplingSec = int(row[0])
        else:
            samplingCount += 1
        
        #check for acceleration
        try:
            accIndex = row.index(datamap.acc[0])
            accXCol = datamap.acc.index('x')
            accYCol = datamap.acc.index('y')
            accZCol = datamap.acc.index('z')
            acceleration[0][0].append(row[accIndex + accXCol])
            acceleration[1][0].append(row[accIndex + accYCol])
            acceleration[2][0].append(row[accIndex + accZCol])
        except ValueError:
            accIndex = None

        #check for gyro
        try:
            gyroIndex = row.index(datamap.gyro[0])
            gyroXCol = datamap.gyro.index('x')
            gyroYCol = datamap.gyro.index('y')
            gyroZCol = datamap.gyro.index('z')
            gyroscope[0][0].append(row[gyroIndex + gyroXCol])
            gyroscope[1][0].append(row[gyroIndex + gyroYCol])
            gyroscope[2][0].append(row[gyroIndex + gyroZCol])
        except ValueError:
            gyroIndex = None

setSample(0)

with open('WillEdit.csv') as file:
    fileobj = csv.reader(file, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for row in fileobj:
        
        if(samplingSec == 0):
            samplingSec = int(row[0])
        
        if(int(row[0]) != samplingSec):
            samplingCulm.append(samplingCount)
            samplingCount = 0
            samplingSec = int(row[0])
        else:
            samplingCount += 1
        
        #check for acceleration
        try:
            accIndex = row.index(datamap.acc[0])
            accXCol = datamap.acc.index('x')
            accYCol = datamap.acc.index('y')
            accZCol = datamap.acc.index('z')
            acceleration[0][1].append(row[accIndex + accXCol])
            acceleration[1][1].append(row[accIndex + accYCol])
            acceleration[2][1].append(row[accIndex + accZCol])
        except ValueError:
            accIndex = None

        #check for gyro
        try:
            gyroIndex = row.index(datamap.gyro[0])
            gyroXCol = datamap.gyro.index('x')
            gyroYCol = datamap.gyro.index('y')
            gyroZCol = datamap.gyro.index('z')
            gyroscope[0][1].append(row[gyroIndex + gyroXCol])
            gyroscope[1][1].append(row[gyroIndex + gyroYCol])
            gyroscope[2][1].append(row[gyroIndex + gyroZCol])
        except ValueError:
            gyroIndex = None

setSample(1)

with open('ChristiaanEdit.csv') as file:
    fileobj = csv.reader(file, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for row in fileobj:
        
        if(samplingSec == 0):
            samplingSec = int(row[0])
        
        if(int(row[0]) != samplingSec):
            samplingCulm.append(samplingCount)
            samplingCount = 0
            samplingSec = int(row[0])
        else:
            samplingCount += 1
        
        #check for acceleration
        try:
            accIndex = row.index(datamap.acc[0])
            accXCol = datamap.acc.index('x')
            accYCol = datamap.acc.index('y')
            accZCol = datamap.acc.index('z')
            acceleration[0][2].append(row[accIndex + accXCol])
            acceleration[1][2].append(row[accIndex + accYCol])
            acceleration[2][2].append(row[accIndex + accZCol])
        except ValueError:
            accIndex = None

        #check for gyro
        try:
            gyroIndex = row.index(datamap.gyro[0])
            gyroXCol = datamap.gyro.index('x')
            gyroYCol = datamap.gyro.index('y')
            gyroZCol = datamap.gyro.index('z')
            gyroscope[0][2].append(row[gyroIndex + gyroXCol])
            gyroscope[1][2].append(row[gyroIndex + gyroYCol])
            gyroscope[2][2].append(row[gyroIndex + gyroZCol])
        except ValueError:
            gyroIndex = None

setSample(2)

with open('LucasEdit.csv') as file:
    fileobj = csv.reader(file, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for row in fileobj:
        
        if(samplingSec == 0):
            samplingSec = int(row[0])
        
        if(int(row[0]) != samplingSec):
            samplingCulm.append(samplingCount)
            samplingCount = 0
            samplingSec = int(row[0])
        else:
            samplingCount += 1
        
        #check for acceleration
        try:
            accIndex = row.index(datamap.acc[0])
            accXCol = datamap.acc.index('x')
            accYCol = datamap.acc.index('y')
            accZCol = datamap.acc.index('z')
            acceleration[0][3].append(row[accIndex + accXCol])
            acceleration[1][3].append(row[accIndex + accYCol])
            acceleration[2][3].append(row[accIndex + accZCol])
        except ValueError:
            accIndex = None

        #check for gyro
        try:
            gyroIndex = row.index(datamap.gyro[0])
            gyroXCol = datamap.gyro.index('x')
            gyroYCol = datamap.gyro.index('y')
            gyroZCol = datamap.gyro.index('z')
            gyroscope[0][3].append(row[gyroIndex + gyroXCol])
            gyroscope[1][3].append(row[gyroIndex + gyroYCol])
            gyroscope[2][3].append(row[gyroIndex + gyroZCol])
        except ValueError:
            gyroIndex = None

setSample(3)

def normalize(a):
    avg = sum(a)/len(a)
    for x in range(0, len(a)):
        a[x] = a[x]-avg
    
    return a

def integrate(a,u):
    culm = 0
    res = []
    for x in range(0, len(a)):
        if(abs(a[x]) < 38):
            culm += a[x]/sampling[u]
            res.append(culm)
        else:
            res.append(culm)
    
    return res

def plot():

    plt.plot(acceleration[0][0])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Nick - Acceleration X")
    plt.xlabel("Sample")
    plt.ylabel("Acceleration (m/s^2)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')
    
    acceleration[1][0] = normalize(acceleration[1][0])
    
    plt.figure()
    plt.plot(integrate(acceleration[0][0], 0))
    plt.ylim([-15,15])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Nick - Velocity X")
    plt.xlabel("Sample")
    plt.ylabel("Velocity (m/s)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(integrate(integrate(acceleration[0][0], 0), 0))
    plt.ylim([-500,100])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Nick - Position X")
    plt.xlabel("Sample")
    plt.ylabel("Position (m)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(acceleration[1][0])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Nick - Acceleration Y")
    plt.xlabel("Sample")
    plt.ylabel("Acceleration (m/s^2)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')
    
    acceleration[1][0] = normalize(acceleration[1][0])
    
    plt.figure()
    plt.plot(integrate(acceleration[1][0], 0))
    plt.ylim([-15,15])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Nick - Velocity Y")
    plt.xlabel("Sample")
    plt.ylabel("Velocity (m/s)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(integrate(integrate(acceleration[1][0], 0), 0))
    plt.ylim([-500,100])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Nick - Position Y")
    plt.xlabel("Sample")
    plt.ylabel("Position (m)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(acceleration[2][0])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Nick - Acceleration Z")
    plt.xlabel("Sample")
    plt.ylabel("Acceleration (m/s^2)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')
    
    acceleration[1][0] = normalize(acceleration[1][0])
    
    plt.figure()
    plt.plot(integrate(acceleration[2][0], 0))
    plt.ylim([-15,15])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Nick - Velocity Z")
    plt.xlabel("Sample")
    plt.ylabel("Velocity (m/s)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(integrate(integrate(acceleration[2][0], 0), 0))
    plt.ylim([-500,100])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Nick - Position Z")
    plt.xlabel("Sample")
    plt.ylabel("Position (m)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(gyroscope[0][0])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Nick - Gyroscope X")
    plt.xlabel("Sample")
    plt.ylabel("Angular Velocity (rad/s))")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(gyroscope[1][0])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Nick - Gyroscope Y")
    plt.xlabel("Sample")
    plt.ylabel("Angular Velocity (rad/s))")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(gyroscope[2][0])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Nick - Gyroscope Z")
    plt.xlabel("Sample")
    plt.ylabel("Angular Velocity (rad/s))")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(acceleration[0][1])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Will - Acceleration X")
    plt.xlabel("Sample")
    plt.ylabel("Acceleration (m/s^2)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')
    
    acceleration[1][0] = normalize(acceleration[1][0])
    
    plt.figure()
    plt.plot(integrate(acceleration[0][1], 0))
    plt.ylim([-15,15])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Will - Velocity X")
    plt.xlabel("Sample")
    plt.ylabel("Velocity (m/s)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(integrate(integrate(acceleration[0][1], 0), 0))
    plt.ylim([-500,100])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Will - Position X")
    plt.xlabel("Sample")
    plt.ylabel("Position (m)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(acceleration[1][1])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Will - Acceleration Y")
    plt.xlabel("Sample")
    plt.ylabel("Acceleration (m/s^2)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')
    
    acceleration[1][0] = normalize(acceleration[1][0])
    
    plt.figure()
    plt.plot(integrate(acceleration[1][1], 0))
    plt.ylim([-15,15])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Will - Velocity Y")
    plt.xlabel("Sample")
    plt.ylabel("Velocity (m/s)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(integrate(integrate(acceleration[1][1], 0), 0))
    plt.ylim([-500,100])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Will - Position Y")
    plt.xlabel("Sample")
    plt.ylabel("Position (m)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(acceleration[2][1])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Will - Acceleration Z")
    plt.xlabel("Sample")
    plt.ylabel("Acceleration (m/s^2)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')
    
    acceleration[1][0] = normalize(acceleration[1][0])
    
    plt.figure()
    plt.plot(integrate(acceleration[2][1], 0))
    plt.ylim([-15,15])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Will - Velocity Z")
    plt.xlabel("Sample")
    plt.ylabel("Velocity (m/s)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(integrate(integrate(acceleration[2][1], 0), 0))
    plt.ylim([-500,100])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Will - Position Z")
    plt.xlabel("Sample")
    plt.ylabel("Position (m)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(gyroscope[0][1])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Will - Gyroscope X")
    plt.xlabel("Sample")
    plt.ylabel("Angular Velocity (rad/s))")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(gyroscope[1][1])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Will - Gyroscope Y")
    plt.xlabel("Sample")
    plt.ylabel("Angular Velocity (rad/s))")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(gyroscope[2][1])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Will - Gyroscope Z")
    plt.xlabel("Sample")
    plt.ylabel("Angular Velocity (rad/s))")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(acceleration[0][2])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Chris - Acceleration X")
    plt.xlabel("Sample")
    plt.ylabel("Acceleration (m/s^2)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')
    
    acceleration[1][0] = normalize(acceleration[1][0])
    
    plt.figure()
    plt.plot(integrate(acceleration[0][2], 0))
    plt.ylim([-15,15])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Chris - Velocity X")
    plt.xlabel("Sample")
    plt.ylabel("Velocity (m/s)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(integrate(integrate(acceleration[0][2], 0), 0))
    plt.ylim([-500,100])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Chris - Position X")
    plt.xlabel("Sample")
    plt.ylabel("Position (m)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(acceleration[1][2])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Chris - Acceleration Y")
    plt.xlabel("Sample")
    plt.ylabel("Acceleration (m/s^2)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')
    
    acceleration[1][0] = normalize(acceleration[1][0])
    
    plt.figure()
    plt.plot(integrate(acceleration[1][2], 0))
    plt.ylim([-15,15])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Chris - Velocity Y")
    plt.xlabel("Sample")
    plt.ylabel("Velocity (m/s)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(integrate(integrate(acceleration[1][2], 0), 0))
    plt.ylim([-500,100])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Chris - Position Y")
    plt.xlabel("Sample")
    plt.ylabel("Position (m)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(acceleration[2][2])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Chris - Acceleration Z")
    plt.xlabel("Sample")
    plt.ylabel("Acceleration (m/s^2)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')
    
    acceleration[1][0] = normalize(acceleration[1][0])
    
    plt.figure()
    plt.plot(integrate(acceleration[2][2], 0))
    plt.ylim([-15,15])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Chris - Velocity Z")
    plt.xlabel("Sample")
    plt.ylabel("Velocity (m/s)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(integrate(integrate(acceleration[2][2], 0), 0))
    plt.ylim([-500,100])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Chris - Position Z")
    plt.xlabel("Sample")
    plt.ylabel("Position (m)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(gyroscope[0][2])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Chris - Gyroscope X")
    plt.xlabel("Sample")
    plt.ylabel("Angular Velocity (rad/s))")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(gyroscope[1][2])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Chris - Gyroscope Y")
    plt.xlabel("Sample")
    plt.ylabel("Angular Velocity (rad/s))")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(gyroscope[2][2])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Chris - Gyroscope Z")
    plt.xlabel("Sample")
    plt.ylabel("Angular Velocity (rad/s))")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(acceleration[0][3])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Luke - Acceleration X")
    plt.xlabel("Sample")
    plt.ylabel("Acceleration (m/s^2)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')
    
    acceleration[1][0] = normalize(acceleration[1][0])
    
    plt.figure()
    plt.plot(integrate(acceleration[0][3], 0))
    plt.ylim([-15,15])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Luke - Velocity X")
    plt.xlabel("Sample")
    plt.ylabel("Velocity (m/s)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(integrate(integrate(acceleration[0][3], 0), 0))
    plt.ylim([-500,100])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Luke - Position X")
    plt.xlabel("Sample")
    plt.ylabel("Position (m)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(acceleration[1][3])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Luke - Acceleration Y")
    plt.xlabel("Sample")
    plt.ylabel("Acceleration (m/s^2)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')
    
    acceleration[1][0] = normalize(acceleration[1][0])
    
    plt.figure()
    plt.plot(integrate(acceleration[1][3], 0))
    plt.ylim([-15,15])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Luke - Velocity Y")
    plt.xlabel("Sample")
    plt.ylabel("Velocity (m/s)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(integrate(integrate(acceleration[1][3], 0), 0))
    plt.ylim([-500,100])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Luke - Position Y")
    plt.xlabel("Sample")
    plt.ylabel("Position (m)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(acceleration[2][3])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Luke - Acceleration Z")
    plt.xlabel("Sample")
    plt.ylabel("Acceleration (m/s^2)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')
    
    acceleration[1][0] = normalize(acceleration[1][0])
    
    plt.figure()
    plt.plot(integrate(acceleration[2][3], 0))
    plt.ylim([-15,15])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Luke - Velocity Z")
    plt.xlabel("Sample")
    plt.ylabel("Velocity (m/s)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(integrate(integrate(acceleration[2][3], 0), 0))
    plt.ylim([-500,100])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Luke - Position Z")
    plt.xlabel("Sample")
    plt.ylabel("Position (m)")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(gyroscope[0][3])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Luke - Gyroscope X")
    plt.xlabel("Sample")
    plt.ylabel("Angular Velocity (rad/s))")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(gyroscope[1][3])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Luke - Gyroscope Y")
    plt.xlabel("Sample")
    plt.ylabel("Angular Velocity (rad/s))")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.figure()
    plt.plot(gyroscope[2][3])
    plt.ylim([-50,50])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Luke - Gyroscope Z")
    plt.xlabel("Sample")
    plt.ylabel("Angular Velocity (rad/s))")
    #plt.annotate('Not So Great', xy=(np.argmin(r_v), np.min(r_v)), xytext=(np.argmin(r_v)-10, 0),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right')

    plt.show()

plot()


# In[ ]:




# In[ ]:



