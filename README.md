# Learned Binary Search
The official implementation of Learned BST data structure and the baselines in the paper 
"Binary Search with Distributional Predictions" 
by Michael Dinitz, Sungjin Im, Thomas Lavastida, Benjamin Moseley, Aidin Niaparast, and Sergei Vassilvitskii.

## Datasets
We use real temporal networks from the Stanford Large Network Dataset Collection: https://snap.stanford.edu/data/

In particular, the following datasets are used:
1. Ask Ubuntu (https://snap.stanford.edu/data/sx-askubuntu.html): file sx-askubuntu-a2q.txt
2. Super User (https://snap.stanford.edu/data/sx-superuser.html): file sx-superuser-a2q.txt
3. Stack Overflow (https://snap.stanford.edu/data/sx-stackoverflow.html): file sx-stackoverflow-a2q.txt

## Codes
1. LBST.py: implements the main algorithm of the paper (Learned BST) and the baselines (Classic and Bisection)
2. Test.py: prints and plots the results of the synthetic and real data experiments

## Reproducing the results
1. For obtaining the results of the synthetic experiment, run the syntheticTest function with the desired range size.
The plot in the paper corresponds to running syntheticTest(200000).
2. For real data experiments, first, the files mentioned above need to be downloaded into a folder named "Datasets".
Then calling the function testRealDataset with the appropriate dataset name will print and plot the results.
