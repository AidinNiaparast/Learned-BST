import math
from sortedcontainers import SortedList
import random

class Node:
    def __init__(self, key, l, r):
        self.key = key
        self.leftChild = None
        self.rightChild = None
        self.l = l #start of node's interval (inclusive)
        self.r = r #end of node's interval (exclusive)

class LearnedBinarySearchTree:
    def __init__(self, keys, predictions):
        self.n = len(keys)
        self.keys = keys #the keys are sorted in increasing order
        self.frequencies = list(predictions) #predictions[i] is the predicted probability of accessing keys[i]
        self.root = None
        self.frequencySum = [0] * self.n #frequencySum[i] stores the sum of predictions from 0 to i (inclusive)
        self.frequencySum[0] = predictions[0]
        for i in range(1,self.n):
            self.frequencySum[i] = self.frequencySum[i-1] + predictions[i]  
        self.sortedFrequencySum = SortedList(self.frequencySum) #is used only for the bisection rule
        self.activeLeaves = [] #stores the set of active leaves of the binary search tree, 
        #which are the ones that have more than one index in their interval and need to be further broken down 
        self.newLeaves = []    

    #Returns the sum of the frequencies from l to r-1
    def getFrequencySum(self,l,r):
        if l == 0:
            return self.frequencySum[r-1]
        return self.frequencySum[r-1] - self.frequencySum[l-1]

    #Builds a classis BST on [l,r), where l=currentNode.l and r=currentNode.r
    def buildClassicBST(self,currentNode):
        l = currentNode.l
        r = currentNode.r
        if l == r:
            return
        if l + 1 == r:
            currentNode.key = self.keys[l]
            return
        mid = (l+r) // 2
        currentNode.key = self.keys[mid]
        currentNode.leftChild = Node(None, l, mid)
        currentNode.rightChild = Node(None, mid+1, r)
        self.buildClassicBST(currentNode.leftChild)
        self.buildClassicBST(currentNode.rightChild)   

    #Builds a BST with depth d with currentNode as the root using the bisection rule.  
    #When the sum of the frequencies in the search range is zero, the function creates 
    #a classic BST on the search range
    def buildBisectionBST(self,currentNode,d):
        l = currentNode.l
        r = currentNode.r
        if l == r:
            return
        if l + 1 == r:
            currentNode.key = self.keys[l]
            return
        if d == 0:
            #this node needs to be further broken down
            self.newLeaves.append(currentNode)
            return
        #if the sum of the frequencies in the search range is zero, it builds a classic BST
        if self.getFrequencySum(l,r) == 0:
            mid = (l+r)//2
        else:
            #want to find the first index i such that frequency[l]+...+frequency[i] >= 1/2(frequency[l]+...+frequency[r])
            if l>0:
                cutOff = self.frequencySum[l-1] + 0.5*self.getFrequencySum(l,r)
            else:
                cutOff = 0.5*self.getFrequencySum(l,r)  
            mid = self.sortedFrequencySum.bisect_left(cutOff)
        currentNode.key = self.keys[mid]
        currentNode.leftChild = Node(None, l, mid)
        currentNode.rightChild = Node(None, mid+1, r)
        self.buildBisectionBST(currentNode.leftChild,d-1)
        self.buildBisectionBST(currentNode.rightChild,d-1)        

    #Performs the "Binary Search at the Endpoints" phase of the Learned BST algorithm with parameter d
    #Builds two classic binary search trees on the intervals [l,l+d) and [r-d,r)
    def buildEndpointBST(self,currentNode,d):
        l = currentNode.l
        r = currentNode.r
        if r-l <= 2 * d + 2:
            self.buildClassicBST(currentNode)
            return
        currentNode.key = self.keys[r-d-1]
        rightChild = Node(None,r-d,r)
        self.buildClassicBST(rightChild)
        leftChild = Node(None,l,r-d-1)
        leftChild.key = self.keys[l+d]
        leftLeftChild = Node(None,l,l+d)
        self.buildClassicBST(leftLeftChild)
        leftRightChild = Node(None,l+d+1,r-d-1)
        currentNode.leftChild = leftChild
        currentNode.rightChild = rightChild
        leftChild.leftChild = leftLeftChild
        leftChild.rightChild = leftRightChild
        currentNode = leftRightChild
        self.newLeaves.append(currentNode)    

    #The main algorithm of the paper that interleaves between BisectionBST and EndpointBST
    def LearnedBST(self):
        self.root = Node(None, 0, self.n)
        self.activeLeaves = [self.root]
        self.newLeaves = []
        i = 0
        while len(self.activeLeaves) > 0:
            #Bisection
            for node in self.activeLeaves: 
                self.buildBisectionBST(node, 2**i)
            self.activeLeaves = list(self.newLeaves)
            self.newLeaves = []    
            #Endpoint Binary Search
            for node in self.activeLeaves:
                self.buildEndpointBST(node, 2 ** (8*(2**i))) #in the paper, this parameter is 2**(2**i)
            self.activeLeaves = list(self.newLeaves)
            self.newLeaves = []     
            i += 1
    
    #Returns the cost of searching for key starting from node 'node'
    def binarySearchCost(self,node,key):
        if node.key == None:
            print('Error: Key ' + str(key) + ' Not Found!')
            quit()
        if node.key == key:
            return 1
        if key < node.key:
            return 1 + self.binarySearchCost(node.leftChild,key)
        else:
            return 1 + self.binarySearchCost(node.rightChild,key)        