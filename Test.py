from sortedcontainers import SortedList
import LBST
from scipy.stats import wasserstein_distance
import numpy as np
import matplotlib.pyplot as plt 
import math
import matplotlib
import statistics

def readDataset(datasetName):
    path = 'Datasets\\'
    fileName = ''
    match datasetName:
        case 'AskUbuntu':
            fileName = 'sx-askubuntu-a2q.txt' 
        case 'SuperUser':
            fileName = 'sx-superuser-a2q.txt'  
        case 'StackOverflow':
            fileName = 'sx-stackoverflow-a2q.txt'
    with open(path + fileName) as f:
        lines = f.readlines()
    data = []    
    #Each line in the dataset contains sourceID, targetID, and a timestamp
    #Sort the dataset in increasing order of timestamp
    lines.sort(key=lambda x: int(x.split()[2]))
    for line in lines:
        source, target, time = line.split()
        #we use the source node as our training and test data
        data.append(int(source))       
    return data  

#This function first replaces each element in data with the largest key in keys that is less than or equal to that element
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
    #project data on keys and get frequencies
    frequencies = {}
    for key in keys:
        frequencies[key] = 0
    for element,count in dataCount.items():
        projectedIndex = max(keys.bisect_right(element)-1,0)
        projectedElement = keys[projectedIndex]
        frequencies[projectedElement] += count   
    #normalize
    for key in keys:
        frequencies[key] /= len(data)
    return frequencies

#Returns the weighted average cost of performing binary search using BST with the test frequencies as weights
def getCost(BST, testFrequencies):
    cost = 0
    for key,value in testFrequencies.items():
        cost += value * BST.binarySearchCost(BST.root,key)
    return cost            

#Returns the Earth Mover's distance between the frequencies obtained from the training data and test data
def getEMD(trainingFrequencies,testFrequencies):
    support = list(range(len(trainingFrequencies)))
    return(wasserstein_distance(support,support,trainingFrequencies,testFrequencies))

#runs the algorithm alg on keys using predictions and returns the weighted cost of alg according to the test frequencies
#alg can be 'LearnedBST', 'ConvexCombination', 'Bisection', or 'Classic'
#LearnedBST: the main algorithm of the paper 
#Classic: the classic BST algorithm that splits the search interval in half each time
#Bisection: it does the Bisection rule until the predicted probability becomes zero. Then it does the Classic rule (splits the interval in half)
#ConvexCombination: performs the Bisection algorithm on a convex combination of the predictions and the uniform distribution
#the convex combination is alpha*uniform + (1-alpha)*predictions
def runAlg(alg, keys, predictions, testFrequencies, alpha=1/2):
    BST = None
    match alg:
        case 'LearnedBST':
            BST = LBST.LearnedBinarySearchTree(keys,predictions)
            BST.LearnedBST()  
        case 'ConvexCombination':
            newFrequencies = [(alpha * 1/len(keys) + (1-alpha) * x) for x  in predictions]
            BST = LBST.LearnedBinarySearchTree(keys,newFrequencies)
            BST.root = LBST.Node(None,0,len(keys))
            BST.buildBisectionBST(BST.root, len(keys))   
        case 'Bisection':
            BST  = LBST.LearnedBinarySearchTree(keys,predictions)
            BST.root = LBST.Node(None,0,len(keys))
            BST.buildBisectionBST(BST.root, len(keys)) #len(keys) is large enough to make sure the whole tree is made using the Bisection rule        
        case 'Classic':
            BST = LBST.LearnedBinarySearchTree(keys,[0]*len(keys))
            BST.root = LBST.Node(None,0,len(keys))
            BST.buildClassicBST(BST.root)
        case _:
            print('Error: invalid algorithm!')
            quit() 
    return getCost(BST,testFrequencies)            

#plots the key frequencies obtained from test and training data
def plotFrequencies(keys, trainingFrequencies,testFrequencies):
    matplotlib.rcParams.update({'font.size': 24})
    xVals = list(range(len(keys))) 
    fig, axs = plt.subplots(2)
    axs[0].bar(xVals, trainingFrequencies, width = 5)
    axs[1].bar(xVals, testFrequencies, width = 5)
    axs[0].set_title('Training')
    axs[1].set_title('Test')
    plt.subplots_adjust(hspace=0.4)
    plt.show()

#this function prints and plots the results of real data experiments explained in the paper
#it compares the costs of the following algorithms: "Learned BST", "Classic", "Bisection", and "Convex Combination" 
#uses the set of elements in the first 10% of the data as the set of keys of the binary search tree 
#uses the first t% of the rest of the data as training data, for t=5,10,...,50, and the rest of it as the test data
def testRealDataset(datasetName):
    print('datasetName=', datasetName)
    data = readDataset(datasetName)
    if len(data) > 1000000:
        data = data[:1000000]
    trainingDataPercentages = [i * 5 for i in range(1,11)]
    keys = sorted(set(data[:len(data)//10]))
    data = data[len(data)//10:]
    dataSize = len(data)
    algs = ['Classic','Bisection','LearnedBST','ConvexCombination']
    costs = {}
    for alg in algs:
        costs[alg] = []
    EMDs = [] #the list of Earth Mover's distances between the predictions and the actual probabilities
    for trainingDataPercentage in trainingDataPercentages:
        trainingDataSize = dataSize * trainingDataPercentage // 100
        trainingData = data[:trainingDataSize]
        testData = data[trainingDataSize:]
        predictionsDict = getFrequencies(keys,trainingData)
        predictions = [predictionsDict[key] for key in keys]
        testFrequenciesDict = getFrequencies(keys,testData)
        testFrequencies = [testFrequenciesDict[key] for key in keys]
        positiveTestFrequencies =  {key:value for key, value in testFrequenciesDict.items() if value > 0}
        #uncomment for the frequency plots
        #if trainingDataPercentage == 50:
        #    plotFrequencies(keys,predictions,testFrequencies)
        #calculate the Earth Mover's distance between the predictions and the actual probability
        EMDs.append(getEMD(predictions,testFrequencies))
        for alg in algs:
            costs[alg].append(runAlg(alg,keys,predictions,positiveTestFrequencies))
    print('EMDs=',EMDs)          
    for alg in algs:
        print(alg + ' costs:',costs[alg])
    matplotlib.rcParams.update({'font.size': 24})
    xVals = [math.log2(x) for x in EMDs]
    markerStyle = "o"
    markerSize = 10
    width = 5
    colors = {'Classic':'blue', 'Bisection':'darkOrange', 'LearnedBST':'green', 'ConvexCombination':'purple'}
    labels = {'Classic':'Classic', 'Bisection':'Bisection', 'LearnedBST':'Learned BST', 'ConvexCombination':'Convex Combination'}
    for alg in algs:
        plt.plot(xVals, costs[alg], color=colors[alg], linestyle='-', label=labels[alg], marker = markerStyle, markersize = markerSize, linewidth = width)
    plt.xlabel('log(Earth Mover\'s Distance)')
    plt.ylabel("Average Cost")  
    loc = 'lower right'
    if datasetName == 'MathOverflow':
        loc = 'center left'
    if datasetName == 'AskUbuntu':
        loc = 'upper left'
    #uncomment for the the plots for cost vs training data percentage 
    #xVals = trainingDataPercentages
    #plt.xlabel("Percentage of Training Data")
    #loc = 'lower left'
    #if datasetName == 'AskUbuntu':  
    #    loc = 'upper right'  
    plt.legend(loc = loc)        
    plt.show()

#this function prints and plots the results of the syntehtic experiment explained in the paper
#it compares the costs of the following algorithms: "Learned BST", "Classic", "Bisection", and "Convex Combination" 
#the training data is a normal distribution with mean 0 and standard deviation 10
#the test data has the same standard deviation, but the mean is shifted
def syntheticTest(rangeSize):
    sampleSize = 10000
    mu = 0
    sigma = 10
    numExperiments = 5 #the training and test sets are regenerated 5 times
    algs = ['Classic','Bisection','LearnedBST','ConvexCombination']
    #mean of the 5 runs
    means = {}
    for alg in algs:
        means[alg] = []
    #standard deviation of the 5 runs
    SDs = {}
    for alg in algs:
        SDs[alg] = []
    keys = list(range(-rangeSize//2,rangeSize//2))
    shifts = []
    costs ={}
    for shift in range(0,351,50):    
        shifts.append(shift)
        for alg in algs:
            costs[alg] = []
        for _ in range(numExperiments):
            trainingData = np.random.normal(mu, sigma, sampleSize)
            predictionsDict = getFrequencies(keys,trainingData)
            predictions = [predictionsDict[key] for key in keys]
            testData = np.random.normal(mu + shift, sigma, sampleSize)
            testFrequenciesDict = getFrequencies(keys,testData)
            positiveTestFrequencies =  {key:value for key, value in testFrequenciesDict.items() if value > 0}
            for alg in algs:
                costs[alg].append(runAlg(alg,keys,predictions,positiveTestFrequencies))
        for alg in algs:
            means[alg].append(statistics. mean(costs[alg]))
            SDs[alg].append(statistics.pstdev(costs[alg]))    
    for alg in algs:
        print(alg + ' means:', means[alg])
        print(alg + 'standard deviations:',SDs[alg])
    matplotlib.rcParams.update({'font.size': 24})
    xVals = shifts
    markerStyle = "o"
    markerSize = 10
    width = 5
    colors = {'Classic':'blue', 'Bisection':'darkOrange', 'LearnedBST':'green', 'ConvexCombination':'purple'}
    labels = {'Classic':'Classic', 'Bisection':'Bisection', 'LearnedBST':'Learned BST', 'ConvexCombination':'Convex Combination'}
    for alg in algs:
        plt.plot(xVals, means[alg], color=colors[alg], linestyle='-', label=labels[alg], marker = markerStyle, markersize = markerSize, linewidth = width)
        plt.fill_between(xVals, (np.array(means[alg]) - np.array(SDs[alg])).tolist(), 
                    (np.array(means[alg]) + np.array(SDs[alg])).tolist(), alpha=0.5,
                    edgecolor=colors[alg], facecolor=colors[alg])
    plt.xlabel('Mean of Test Data')
    plt.ylabel("Average Cost")
    plt.legend(loc = 'upper left')        
    plt.show()


def main():
    #synthetic experiment
    for rangeSize in [2000,20000,200000]:
        print('rangeSize=',rangeSize)
        syntheticTest(rangeSize)
    #real data experiments
    realDatasets = ['AskUbuntu', 'SuperUser', 'StackOverflow']
    for datasetName in realDatasets:
        testRealDataset(datasetName)       

if __name__ == "__main__":
    main()