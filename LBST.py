import math
from sortedcontainers import SortedList

#Learned Binary Search Tree

class Node:
    def __init__(self, key, l, r):
        self.key = key
        self.leftChild = None
        self.rightChild = None
        #the interval in the array that corresponds to this node's subtree is [l,r)
        self.l = l
        self.r = r

class LearnedBinarySearchTree:
    def __init__(self, keys, predictions):
        self.n = len(keys) #the number of keys
        self.keys = keys #the keys are sorted in increasing order
        self.frequencies = predictions #predictions[i] is the predicted probability or frequency of accessing keys[i]
        self.root = None #root of the binary search tree
        self.frequencySum = [0] * self.n #frequencySum[i] stores the sum of predictions from 0 to i (inclusive)
        self.frequencySum[0] = predictions[0]
        for i in range(1,self.n):
            self.frequencySum[i] = self.frequencySum[i-1] + predictions[i]  
        self.sortedFrequencySum = SortedList(self.frequencySum) #used for the bisection rule
        self.activeLeaves = [] #stores the set of active leaves of the binary search tree, 
        #which are the ones that have more than one index in their interval and need to be further broken down 
        self.newLeaves = []   

    def __str__(self):
        ret = self.recursivePrint(self.root, 0)
        #ret = self.recursiveGraphicalPrint(self.root,0)
        return ret

    #used for printing the BST
    def recursiveGraphicalPrint(self,node,space):
        if node is None:
            return ''
        space += 5
        ret = ''
        ret += self.recursiveGraphicalPrint(node.rightChild,space)
        ret += '\n'
        for i in range(5,space):
            ret += '  '
        ret += str(node.key)
        ret += self.recursiveGraphicalPrint(node.leftChild,space)  
        return ret  

    #used for printing the BST
    def recursivePrint(self,node,space):  
        ret = ''
        for _ in range(space):
            ret += ' '
        ret += ' [' + str(node.l) + ',' + str(node.r) + ')'
        ret += ' ' + str(node.key)
        ret += '\n'
        if node.rightChild is not None:    
            ret += self.recursivePrint(node.rightChild, space + 10)
        if node.leftChild is not None:
            ret += self.recursivePrint(node.leftChild, space + 10) 
        return ret

    #Returns self.freqiencies[l]+...+self.frequencies[r-1]
    def getFrequencySum(self,l,r):
        if l == 0:
            return self.frequencySum[r-1]
        return self.frequencySum[r-1] - self.frequencySum[l-1]

    #Builds a classis BST on keys[currentNode.l:currentNode.r]
    def buildClassicBST(self,currentNode):
        l = currentNode.l
        r = currentNode.r
        if l == r: #the interval is empty
            return
        if l + 1 == r: #the interval contains only one key
            currentNode.key = self.keys[l]
            return
        mid = (l+r) // 2
        currentNode.key = self.keys[mid]
        currentNode.leftChild = Node(None, l, mid)
        currentNode.rightChild = Node(None, mid+1, r)
        self.buildClassicBST(currentNode.leftChild)
        self.buildClassicBST(currentNode.rightChild)   

    #Builds a BST with depth d with currentNode as the root using the bisection rule.  
    #When the sum of the frequencies in the search range gets below the threshold, the function creates 
    #a classic BST on the search range
    def buildBisectionBST(self,currentNode,d,threshold=0):
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
        #if the sum of the frequencies in the search range is less than the threshold it builds the classic BST
        if self.getFrequencySum(l,r) <= threshold:
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
        self.buildBisectionBST(currentNode.leftChild,d-1,threshold)
        self.buildBisectionBST(currentNode.rightChild,d-1,threshold)        

    #Builds a BST using the Distance Doubling rule up to distance d
    def buildDistanceDoublingBST(self,currentNode,d):
        dist = round(math.log2(self.n))
        #dist = 1
        while dist <= d:
            l = currentNode.l
            r = currentNode.r
            if r-l <= 2 * dist + 2:
                self.buildClassicBST(currentNode)
                return
            currentNode.key = self.keys[r-dist-1]
            rightChild = Node(None,r-dist,r)
            self.buildClassicBST(rightChild)
            leftChild = Node(None,l,r-dist-1)
            leftChild.key = self.keys[l+dist]
            leftLeftChild = Node(None,l,l+dist)
            self.buildClassicBST(leftLeftChild)
            leftRightChild = Node(None,l+dist+1,r-dist-1)
            currentNode.leftChild = leftChild
            currentNode.rightChild = rightChild
            leftChild.leftChild = leftLeftChild
            leftChild.rightChild = leftRightChild
            currentNode = leftRightChild
            dist *= 2   
        self.newLeaves.append(currentNode)    

    #bisection + distance doubling rule
    #Alg1 is a heuristic based on the algorithm in section 2 that works for EMD. 
    #The algorithm alternates between bisection and distance doubling rules.
    def Alg1(self):
        self.root = Node(None, 0, self.n)
        self.activeLeaves = [self.root]
        self.newLeaves = []
        i = 0
        while len(self.activeLeaves) > 0:
            rho = 2 ** (2 ** i)
            #Bisection
            for node in self.activeLeaves: 
                self.buildBisectionBST(node, 2**i)
            self.activeLeaves = list(self.newLeaves)
            self.newLeaves = []    
            #Distance Doubling
            for node in self.activeLeaves:
                self.buildDistanceDoublingBST(node, rho * math.log2(self.n)) #in the paper the second argument of the function is rho instead of rho * log(n)
            self.activeLeaves = list(self.newLeaves)
            self.newLeaves = []     
            i += 1

    def Alg2(self):
        self.root = Node(None,0,self.n)
        self.buildBisectionBST(self.root, self.n, 10/self.n) #the depth of n is to make sure the whole tree gets built.
        #in the paper, the threshold is 1/n instead of 10/n

    #Returns the cost of searching for key starting from node 'node'
    def binarySearchCost(self,node,key):
        if node.key == None:
            print('Error: Key Not Found!',key)
            quit()
        if node.key == key:
            return 1
        if key < node.key:
            return 1 + self.binarySearchCost(node.leftChild,key)
        else:
            return 1 + self.binarySearchCost(node.rightChild,key)

def main():
    n = 100
    keys = list(range(n))
    predictions = [0] * n
    for i in range(n):
        #predictions[i] = i/n
        predictions[i] = 1/n     
    LBST = LearnedBinarySearchTree(keys,predictions)
    LBST.Alg1()
    print(LBST)

if __name__ == "__main__":
    main()              