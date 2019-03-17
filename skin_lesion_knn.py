import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import scikitplot as skplt
import matplotlib.pyplot as plt

#Load data and change the label of the negative class
set = np.genfromtxt("Features_labels_update.txt")
setpos = np.copy(set[set[:,-1]>0,:])
setneg = np.copy(set[set[:,-1]==0,:])

#First batch
apos = setpos[0:10,:]
aneg = setneg[0:30,:]
a = np.vstack((apos,aneg))
np.random.shuffle(a)
afv = a[:,:-1]
alab = a[:,-1]

#Second batch
bpos = setpos[10:20,:]
bneg = setneg[30:60,:]
b = np.vstack((bpos,bneg))
np.random.shuffle(b)
bfv = b[:,:-1]
blab = b[:,-1]

#Third batch
cpos = setpos[20:30,:]
cneg = setneg[60:90,:]
c = np.vstack((cpos,cneg))
np.random.shuffle(c)
cfv = c[:,:-1]
clab = c[:,-1]

#Fourth batch
dpos = setpos[30:40,:]
dneg = setneg[90:120,:]
d = np.vstack((dpos,dneg))
np.random.shuffle(d)
dfv = d[:,:-1]
dlab = d[:,-1]

#Fifth batch
epos = setpos[40:,:]
eneg = setneg[120:,:]
e = np.vstack((epos,eneg))
np.random.shuffle(e)
efv = e[:,:-1]
elab = e[:,-1]

#Define training, testing and validation sets (feature vectors and labels)
fvtrain = np.vstack((cfv,dfv))
fvtrain = np.vstack((fvtrain,efv))
fvvalid = afv
fvtest = bfv
labtrain = np.concatenate((clab,dlab))
labtrain = np.concatenate((labtrain,elab))
labvalid = alab
labtest = blab
scaler = StandardScaler()
fvtrain = scaler.fit_transform(fvtrain)
fvvalid = scaler.transform(fvvalid)
fvtest = scaler.transform(fvtest)

AUCs = np.zeros(10)
neighbors = np.zeros(10)
for i in range (10):
    current_n = i+10
    neighbors[i] = current_n
    #Training phase
    knn = KNeighborsClassifier(n_neighbors=current_n)
    knn.fit(fvtrain, labtrain) 
    #Validation phase to find optimal number of neighbors
    prediction = knn.predict_proba(fvvalid)
    AUCs[i] = roc_auc_score(labvalid, prediction[:,-1])

#Testing phase
index = np.argmax(AUCs)
optimal_n = neighbors[index]
knn = KNeighborsClassifier(n_neighbors=int(optimal_n)) 
knn.fit(fvtrain, labtrain) 

#Evaluation of the AUC on the test set
test_pred = knn.predict_proba(fvtest)
AUC_test = roc_auc_score(labtest, test_pred[:,-1])

#Plotting ROC-curve
skplt.metrics.plot_roc(labtest, test_pred)
plt.show()

print(optimal_n)
print(AUC_test)

