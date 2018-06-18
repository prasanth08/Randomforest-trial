import numpy as np
import matplotlib.pyplot as plt
import math

f= open("./data/gisette_train.data")
data=[]
for row in f.readlines():
    data.append((row.strip()).split(" "))
f.close()

f= open("./data/gisette_train.labels")
classes=[]
for row in f.readlines():
    classes.append((row.strip()).split(" "))
f.close()

data=np.array(data).astype(int)
classes= np.array(classes).astype(int)
classes=classes[:,0]

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

class node(object):
    def __init__(self, is_leaf, split_var, split_val, label, children):
        self.is_leaf=is_leaf
        self.split_var=split_var
        self.label=label
        self.children=children
        self.split_val=split_val
        
def make_Tree(data_idx, feature_idx):
    global data, classes, features, data_len
    
    if len(data[data_idx]) == 0 :
        print(data_idx)
        return node(None,None,None,None,None)
    
    if len(np.unique(classes[data_idx])) == 1 :
        return node(True, None, None, classes[data_idx][0], None)
    
    if len(unique_rows( data[data_idx.reshape((len(data_idx),1)), feature_idx] )) == 1 :
        (y,c) = np.unique(classes[data_idx],return_counts=True)
        return node(True, None, None, y[np.argmax(c)], None)
    
    sY=classes[data_idx]
    uY= np.unique(sY)
    info_parent=0.0    
    for y in uY:
        info_parent = info_parent + (float(len(sY[sY==y]))/len(sY))**2    
    info_parent=1-info_parent
    
    if info_parent < entropy_threshold :
        (y,c) = np.unique(classes[data_idx],return_counts=True)
        return node(True, None, None, y[np.argmax(c)], None)
    del sY
    del uY
    del info_parent

    info_child = np.zeros((features,), dtype=float)
    split_value = np.zeros((features,), dtype=int)
    
    for j in range(features):
        uX = np.unique(data[data_idx.reshape((len(data_idx),1)) ,feature_idx[j]])
        
        if len(uX) > split_value_thresholds:
            uX = np.random.choice(uX, size=10, replace=False)
            
        temp_value = np.zeros((len(uX),), dtype=float)
        
        for x in range(len(uX)):
            indices=data[data_idx.reshape((len(data_idx),1)),feature_idx[j]]<=uX[x]
            sY=classes[data_idx]
            sY= sY[indices[:,0]]
            uY= np.unique(sY)
            temp_info=0.0
            for y in uY:
                temp_info += (float(len(sY[sY==y]))/len(sY))**2
            temp_info=1-temp_info
            temp_value[x] += len(sY)*(temp_info/len(data_idx))
            
            indices=data[data_idx.reshape((len(data_idx),1)),feature_idx[j]]>uX[x]
            sY=classes[data_idx]
            sY= sY[indices[:,0]]
            uY= np.unique(sY)
            temp_info=0.0
            for y in uY:
                temp_info += (float(len(sY[sY==y]))/len(sY))**2
            temp_info=1-temp_info
            temp_value[x] += len(sY)*(temp_info/len(data_idx))
        info_child[j] = np.min(temp_value)
        split_value[j] = uX[np.argmin(temp_value)]
        del temp_value
        
    optimal_split= feature_idx[np.argmin(info_child)]
    split_label= split_value[np.argmin(info_child)]
    
    del info_child
    del split_value
    
    children=[]
    temp_idx = np.where(data[:,optimal_split] <= split_label)
    mask = np.in1d(  data_idx, temp_idx  )
    temp_data_idx = data_idx[mask]
    
    child = make_Tree(temp_data_idx,feature_idx)
    children.append(child)
    
    temp_idx = np.where(data[:,optimal_split] > split_label)
    mask = np.in1d(  data_idx, temp_idx  )
    temp_data_idx = data_idx[mask]
    
    child = make_Tree(temp_data_idx,feature_idx)
    children.append(child)

    return node(False, optimal_split,split_label,None,children)

def pred_tree(toor, X):
    Y=[]
    for i in range(len(X)):
        temp=toor
        while temp.is_leaf == False:
            if X[i,temp.split_var]<= temp.split_val:
                temp= temp.children[0]
            else:
                temp = temp.children[1]
        Y.append(temp.label)
    return Y

def classification(trees, data):
    t_prediction = np.zeros((len(data), len(trees))).astype(int)
    prediction = np.zeros((len(data),))
    for t in range(len(trees)):
        t_prediction[:,t]  =  np.array(pred_tree(trees[t],data))
    for t in range(len(data)):
        (y,c) = np.unique(t_prediction[t,:],return_counts=True)
        prediction[t] = y[np.argmax(c)]
    return prediction

data_len= len(data)
feature_len= len(data[0])

entropy_threshold = 0.01
split_value_thresholds = 100

root_list=[]
trees= 100
features=int(math.floor(math.sqrt(feature_len)))

for t in range(trees):
    feature_idx = np.random.choice(range(feature_len),size=features,replace=False)
    data_idx = np.random.choice(range(data_len), size=data_len,replace=True)
    root_list.append(make_Tree(data_idx,feature_idx))

data_len= len(data)
accuracy=[]
for t in range(trees):
    prediction =classification(root_list[:t+1], data)
    (y,c) =np.unique(prediction==classes,return_counts=True)
    if len(c)==1:
        accuracy.append(float(c[0])/data_len)
    else:
        accuracy.append(float(c[1])/data_len)
    del y
    del c

plt.plot(range(1,trees+1),accuracy, "r--")
plt.xlabel("Random Forest Size")
plt.ylabel("Train Accuracy")
plt.title("Random Forest with entropy threshold: 0.01")          
plt.show()

del data
del classes

f= open("./data/gisette_valid.data")
data=[]
for row in f.readlines():
    data.append((row.strip()).split(" "))
f.close()

f= open("./data/gisette_valid.labels")
classes=[]
for row in f.readlines():
    classes.append((row.strip()).split(" "))
f.close()

data=np.array(data).astype(int)
classes= np.array(classes).astype(int)
classes=classes[:,0]

accuracy=[]
data_len= len(data)
for t in range(trees):
    prediction =classification(root_list[:t+1], data)
    (y,c) =np.unique(prediction==classes,return_counts=True)
    if len(c)==1:
        accuracy.append(float(c[0])/data_len)
    else:
        accuracy.append(float(c[1])/data_len)
    del y
    del c

plt.plot(range(1,trees+1),accuracy, "r--")
plt.xlabel("Random Forest Size")
plt.ylabel("Validation Accuracy")
plt.title("Random Forest with entropy threshold: 0.01")          
plt.show()

del data
del classes

del root_list