from sortedcontainers import SortedList
import LBST
from scipy.stats import wasserstein_distance
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt 

def readDataset(datasetName):
    path = 'Datasets\\'
    fileName = ''
    match datasetName:
        case 'CollegeMsg':
            fileName = 'CollegeMsg.txt'
        case 'email-Eu-core': 
            fileName = 'email-Eu-core-temporal.txt'
        case 'MathOverflow':
            fileName = 'sx-mathoverflow-a2q.txt'
    with open(path + fileName) as f:
        lines = f.readlines()
    #Each line in the data contains sourceID, targetID, and a timestamp
    #Sort the dataset in increasing order of timestamp
    lines.sort(key=lambda x: int(x.split()[2]))
    data = []
    for line in lines:
        source, target, time = line.split()
        #we use the source node as our training and test data
        data.append(int(source))       
    return data  


def barPlot(data):
    keys = list(set(data))
    frequencies = [data.count(key) for key in keys]
    #used for visualizing the Zipf distribution. Only the first few keys have noticeable frequencies
    plt.bar(keys[:20], frequencies[:20], color ='maroon', width = 0.4)
    plt.xlabel("Keys")
    plt.ylabel("Frequencies")
    plt.title("Zipf Distribution")
    plt.show()

def scatterPlot(data,dataSetName):
    plt.scatter(list(range(len(data))),list(data))
    plt.xlabel('Time')
    plt.ylabel('Element')
    plt.title(dataSetName)
    plt.show()

#First rounds down each element in data to the nearest key in keys
#Then finds the frequencies of each key and normalizes the frequencies to sum up to one
#Returns the frequencies as a dictionary of the form key:frequency
def getFrequencies(keys,data):
    keys = SortedList(keys)
    #calculate the number of repetitions of each data point
    dataCount = {}
    for element in data:
        if element not in dataCount:
            dataCount[element] = 1
        else:
            dataCount[element] += 1    
    #round data and get frequencies
    frequencies = {}
    for key in keys:
        frequencies[key] = 0
    for element,count in dataCount.items():
        roundedIndex = max(keys.bisect_right(element)-1,0)
        roundedElement = keys[roundedIndex]
        frequencies[roundedElement] += count   
    #normalize
    for key in keys:
        frequencies[key] /= len(data)
    return frequencies

#Returns the weighted average cost of performing binary search using BST with the frequencies as weights
#testFrequencies only contains the elements with positive frequencies
def getCost(BST, testFrequencies):
    cost = 0
    for key,value in testFrequencies.items():
        cost += value * BST.binarySearchCost(BST.root,key)
    return cost            

#Returns the earth mover's distance between the frequencies obtained by the training data and test data
#trainingFrequencies and testFrequencies are both lists
def getEMD(trainingFrequencies,testFrequencies):
    support = list(range(len(trainingFrequencies)))
    return(wasserstein_distance(support,support,trainingFrequencies,testFrequencies))

#Returns the total variation distance between the frequencies obtained by the training data and test data
#trainingFrequencies and testFrequencies are both lists
def getTV(trainingFrequencies,testFrequencies):
    return 0.5 * np.linalg.norm(np.array(trainingFrequencies)-np.array(testFrequencies), ord=1)

#Returns the cross entropy of predictions (training frequencies) relative to the actual distribution (test frequencies)
#this is the cost of the bisection BST when the two distributions have the same support
#trainingFrequencies and testFrequencies are both lists
def getCE(trainingFrequencies,testFrequencies):
    testFrequencies = np.array(testFrequencies)
    trainingFrequencies = np.array(trainingFrequencies)
    mask = trainingFrequencies > 0
    testFrequencies = testFrequencies[mask]
    trainingFrequencies = trainingFrequencies[mask]    
    ce = -np.sum(testFrequencies * np.log(trainingFrequencies)) 
    return ce

#alg can be 'Alg1', 'Alg2', 'Bisection', or 'Classic'
#Alg1: a heuristic based on the algorithm in section 2 that works for EMD. 
#The algorithm alternates between bisection and distance doubling rules.
#Alg2: the algorithm in section 4 that works for TV. 
#This algorithm does the bisection rule until the predicted probability becomes low. Then it does the classic rule.
#Bisection: it does the bisection rule until the predicted probability becomes zero. Then it does the classic rule. The cost is almost H(p,\hat(p))
def runAlg(alg, keys, predictions, testFrequencies):
    BST = None
    startTime = time.time()
    match alg:
        case 'Alg1':
            BST = LBST.LearnedBinarySearchTree(keys,predictions)
            BST.Alg1()
            print('Alg1 building done in ', time.time() - startTime, ' seconds!')
        case 'Alg2':
            BST = LBST.LearnedBinarySearchTree(keys,predictions)
            BST.Alg2()
            print('Alg2 building done in ', time.time() - startTime, ' seconds!')    
        case 'Bisection':
            BST  = LBST.LearnedBinarySearchTree(keys,predictions)
            BST.root = LBST.Node(None,0,len(keys))
            BST.buildBisectionBST(BST.root, len(keys)) #len(keys) is large enough to make sure the whole tree is made using the bisection rule 
            print('Bisection building done in ', time.time() - startTime, ' seconds!')        
        case 'Classic':
            BST = LBST.LearnedBinarySearchTree(keys,[0]*len(keys))
            BST.root = LBST.Node(None,0,len(keys))
            BST.buildClassicBST(BST.root)
            print('Classic building done in ', time.time() - startTime, ' seconds!')
        case _:
            print('Error: invalid algorithm!')
            quit() 
    return getCost(BST,testFrequencies)        


#uses the first half of the data as training data and the second half as test data
#sets an initial set of keys using the first half of the training data, and then scales
#the range of keys according to the pattern seen in the first half of training set (using the best fit line)
def testRealDataset(datasetName):
    print('datasetName=', datasetName)
    data = readDataset(datasetName)
    dataSize = len(data)
    trainingDataPercentages = [i * 5 for i in range(1,11)]
    
    keysSizes = []
    Alg1Costs = []
    Alg2Costs = []
    classicCosts = []
    bisectionCosts = []
    EMDs = []
    TVs = []
    CEs = []

    for trainingDataPercentage in trainingDataPercentages:
        trainingDataSize = dataSize * trainingDataPercentage // 100
        trainingData = data[:trainingDataSize]
        testData = data[trainingDataSize:]
        keys = list(set(trainingData[:trainingDataSize//2]))
        keys.sort()
        #fitting a line to the first half of the training data
        slope, intercept = np.polyfit(list(range(trainingDataSize//2)), trainingData[:trainingDataSize//2], 1)
        addedKeys = []
        for key in keys:
            for i in range(1,dataSize//trainingDataSize):
                addedKeys.append(key + slope * i * trainingDataSize) 
        keys += addedKeys
        keys = list(set(keys))
        keys.sort()
        #scatterPlot(keys,datasetName)
        keysSizes.append(len(keys))

        predictionsDict = getFrequencies(keys,trainingData[trainingDataSize//2:])
        predictions = [predictionsDict[key] for key in keys]

        testFrequenciesDict = getFrequencies(keys,testData)
        testFrequencies = [testFrequenciesDict[key] for key in keys]
        positiveTestFrequencies =  {key:value for key, value in testFrequenciesDict.items() if value > 0}
        
        #calculate the distance between the predictions and the actual probabilities according to different metrics
        EMDs.append(getEMD(predictions,testFrequencies))
        TVs.append(getTV(predictions,testFrequencies))
        CEs.append(getCE(predictions,testFrequencies))
        
        #Alg1
        Alg1Costs.append(runAlg('Alg1',keys,predictions,positiveTestFrequencies))
        #Alg2
        Alg2Costs.append(runAlg('Alg2',keys,predictions,positiveTestFrequencies))
        #bisection
        bisectionCosts.append(runAlg('Bisection',keys,predictions,positiveTestFrequencies))
        #classic
        classicCosts.append(runAlg('Classic',keys,predictions,positiveTestFrequencies)) #it does not use the predictions

    print('EMDs=',EMDs)   
    print('TVs=',TVs)
    print('CEs=',CEs)        
    print('Alg1Costs=', Alg1Costs)    
    print('Alg2Costs=',Alg2Costs)
    print('bisectionCosts=',bisectionCosts)
    print('classicCosts=',classicCosts)

    numOfRows = len(trainingDataPercentages)
    df = pd.DataFrame({'Dataset Size': [len(data)] * numOfRows, 'Training Data %': trainingDataPercentages, 'n':keysSizes, 
                       'EMD':EMDs, 'TV':TVs,
                        'Alg1': Alg1Costs, 'Alg2':Alg2Costs, 'Classic BST': classicCosts, 'Bisection BST': bisectionCosts})
    with pd.ExcelWriter('Results.xlsx', engine='openpyxl', mode='a', if_sheet_exists='new') as writer:    
        df.to_excel(writer, sheet_name=datasetName, index=False)

def syntheticTest():
    sampleSize = 50000
    
    #Gaussian distributions for training and test
    #mu = 0
    #sigma = 10
    #trainingData = np.random.normal(mu, sigma, sampleSize)
    
    #Zipf distribution
    a = 4
    trainingData = np.random.zipf(a, sampleSize)
    #barPlot(list(trainingData))
    
    rangeSize = 100000
    rangeStart = -rangeSize
    rangeEnd = rangeSize
    keys = list(range(rangeStart,rangeEnd))
    predictionsDict = getFrequencies(keys,trainingData)
    predictions = [predictionsDict[key] for key in keys]
    #print('len(keys)=',len(keys))
    
    Alg1Costs = []
    Alg2Costs = []
    classicCosts = []
    bisectionCosts = []
    shifts = []
    EMDs = []
    TVs = []
    CEs = []   

    for shift in range(0,21,2):
        print('shift=',shift)
        shifts.append(shift)
        #Gaussian distribution
        #testData = np.random.normal(mu + shift, sigma, sampleSize)
        #Zipf distribution
        testData = np.random.zipf(a, sampleSize) + shift
        testFrequenciesDict = getFrequencies(keys,testData)
        testFrequencies = [testFrequenciesDict[key] for key in keys]
        positiveTestFrequencies =  {key:value for key, value in testFrequenciesDict.items() if value > 0}
        EMDs.append(getEMD(predictions,testFrequencies))
        TVs.append(getTV(predictions,testFrequencies))
        CEs.append(getCE(predictions,testFrequencies))
        #Alg1
        Alg1Costs.append(runAlg('Alg1',keys,predictions,positiveTestFrequencies))
        #Alg2
        Alg2Costs.append(runAlg('Alg2',keys,predictions,positiveTestFrequencies))
        #bisection
        bisectionCosts.append(runAlg('Bisection',keys,predictions,positiveTestFrequencies))
        #classic
        classicCosts.append(runAlg('Classic',keys,predictions,positiveTestFrequencies)) #it does not use the predictions
    print('EMDs=',EMDs)   
    print('TVs=',TVs)
    print('CEs=',CEs)       
    print('Alg1Costs=', Alg1Costs)    
    print('Alg2Costs=',Alg2Costs)
    print('bisectionCosts=',bisectionCosts)
    print('classicCosts=',classicCosts)

    numOfRows = len(shifts)
    '''
    #normal distribution
    df = pd.DataFrame({'Keys': ['('+str(rangeStart)+','+str(rangeEnd)+')'] * numOfRows, 'mean of test data':shifts, 'std of test data':[sigma] * numOfRows,
                        'EMD':EMDs, 'TV':TVs,
                        'Alg1': Alg1Costs, 'Alg2':Alg2Costs, 'Bisection BST': bisectionCosts, 'Classic BST': classicCosts,
                        })
    with pd.ExcelWriter('Synthetic.xlsx', engine='openpyxl', mode='a', if_sheet_exists='new') as writer:    
        df.to_excel(writer, sheet_name='Normal-'+str(rangeSize), index=False)
    '''
    #zipf distributions
    df = pd.DataFrame({'Keys': ['('+str(rangeStart)+','+str(rangeEnd)+')'] * numOfRows, 'shift of test data':shifts,
                        'EMD':EMDs, 'TV':TVs,
                        'Alg1': Alg1Costs, 'Alg2':Alg2Costs, 'Bisection BST': bisectionCosts, 'Classic BST': classicCosts,
                        })
    with pd.ExcelWriter('Synthetic.xlsx', engine='openpyxl', mode='a', if_sheet_exists='new') as writer:    
        df.to_excel(writer, sheet_name='Zipf-'+str(rangeSize)+',a='+str(a), index=False)



def main():
    for datasetName in ['CollegeMsg','email-Eu-core','MathOverflow']:
        data = readDataset(datasetName)
        print('len(data)=',len(data))
        scatterPlot(data,datasetName)
        testRealDataset(datasetName)
    #syntheticTest()
    

if __name__ == "__main__":
    main()

