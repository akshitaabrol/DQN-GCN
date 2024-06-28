# DQN-GCN
List of input data to send to AI for a 5 node network:
1.	7 features in form of numpy array (5*7). (Can include the preprocessing logic in the test file itself in the function process_data())
Details of 7 features:
a.	Node degree normalized by dividing with max node degree
b.	Node importance 
c.	Tx normalized by dividing with max link capacity
d.	Rx normalized by dividing with max link capacity
e.	Source node converted to one hot encoder
f.	Destination node converted to one hot encoder
g.	Requested bandwidth normalized by dividing with max link capacity
2.	Source node in the form of a number (eg 1 or 2 or 3 or 4 or 5)
3.	Destination node in the form of a number
4.	A_norm (save this as numpy array named adjacency_matrix.npy)
 
Output from AI:
Path 0-59 which needs to be mapped back to the paths in controller. 
Eg. When Source = Switch 1 and Destination = Switch 3 and Path = First Shortest Path (0), AI will return 3.
 

Note: Dummy data is already included in the file for understanding
