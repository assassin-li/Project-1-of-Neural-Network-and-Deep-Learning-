Babysitting the Learning Process

## Three Step Process

· Use needs to define metric-based goals

· Build an end-to-end system

Data-driven refinement

## How to prepare the data?

Step 1: Preprocess the data  
<!-- image-->

<!-- image-->  
(Assume X [NxD] is data matrix, each example in a row)

<!-- image-->

## Which model to use?

Step 2: Choose the architecture: say we start with one hidden layer of 50 neurons:

<!-- image-->

## Choose Metrics

· Accuracy? (% of examples correct)

· Coverage? (% of examples processed)

· Precision? (% of detections that are right)

· Recall? (% of objects detected)

· Amount of error? (For regression problems)

## Is the Loss Reasonable?(1)

## Double check that the loss is reasonable:

definit_two_layer_model(input_size,hidden_size,output_size): #initialize a model model={ model[wi']=0.ooo1\*np.random.randn(input_size,hidden_size) model['bl']=np.zeros(hidden_size) model['w2']=0.oool\*np.random.randn(hidden_size,output_size) model['b2']=np.zeros(output_size) return model

loss \~2.3. "correct for 10 classes

returns the loss and the gradient for all parameters

## Is the Loss Reasonable?(2)

definit_two_layer_model(input_size,hidden_size,output_size): # initialize a model model={} model['wi']=0.ooo1\*np.random.randn(input_size,hidden_size) model['bl']=np.zeros(hidden_size) model['w2']=0.ooo1\*np.random.randn(hidden_size,output_size) model['b2']=np.zeros(output_size) return model

model=init_two_layer_model(32\*32\*3,50,10） #input hidden size,number of classes  
loss,grad=two_layer net(X_train，model,y_train,1e3) crank up regularization  
print ioss

## Then, let's try to train it.

Lets try to train now...

Tip: Make sure that you can overfit very small portion of the training data

model=init_two layer model(32\*32\*3,50，10） #input size,hidden size,number of classes  
trainer=ClassifierTrainer()  
X_tiny=X_train[:20] #take 20 examples  
y_tiny=y_train[:20]  
best_model,stats=trainer.train(X_tiny，y_tiny，X_tiny，y_tiny,  
model,two_layer_net,  
num_epochs=2ee,reg=0.0,  
update='sgd'，learning_rate_decay=1,  
sample batches=False,  
learning_rate=le-3,verbose=True)

## The above code:

take the first 20 examples from CIFAR-10

turn off regularization (reg = 0.0)

use simple vanilla ‘sgd'

## Lets try to train now...

## Tip: Make sure that you can overfit very small portion of the training data

## Very small loss, train accuracy 1.00, nice!

```csv
model=init_two layer model(32*32*3,50，10）#input size，hidden size,number of classes
trainer=ClassifierTrainer()
X_tiny=X_train[:20]#take20 examples
ytiny=y_train[:20]
bestmodel，stats=trainer.train(x_tiny，y_tiny，X_tiny，y_tiny,
model,two_layer_net,
num_epochs=20e,reg=0.0,
update='sgd',learning_rate_decay=1,
sample batches=False,
learning_rate=le-3,verbose=True)
Finished ep0ch 1/200:cost 2.302603，train:0.400000，val 0.400000，lr1.000000e-03 二
Finished ep0ch2/200:c0st2.302258,train:0.450000，val0.450000，Lr1.000000e-03
Finished ep0ch3/200:c0st 2.301849,train:0.600000,val0.600000，1r1.00000e-03
Finishedep0ch4/200:cost2.301196，train:0.650000，val0.65000，Lr1.000000e-03
Finished ep0ch5/200:c0st 2.300044,train:0.650000,val0.650000，Lr1.000000e-03
Finished ep0ch6/200:c0st2.297864,train:0.550000,val0.550000，1r1.000000e-03
Finishedep0ch7/200:Cost2.293595,train:0.6000,val0.600000,1r1.00000e-03
Finished ep0ch8/200:c0st 2.285096，train:0.550000，val0.550000，lr1.000000e-03
Finished ep0ch9/200:cost2.268094,train:0.55000，val0.550000,1r1.000000e-03
Finished ep0ch10/200:c0st2.234787,train:0.500000,val0.500000，lr1.000000e-03
Finished epoch11/200:c0st 2.173187，train:0.50000,val0.500000,lr1.000000e-03
Finished ep0ch12/200:c0st2.076862，train:0.50000，val0.50000，r1.00000e-03
Finishedep0ch13/200:cost1.974090,train:0.400000,val0.40000，1r1.000000e-03
Finished ep0ch14/200:c0st1.895885,train:0.40000,val0.400000,1r1.000000e-03
Finishedep0ch15/200:c0st1.820876,train:0.45000,val0.450000,1r1.000000e-03
Finished ep0ch16/200:c0st1.737430，train:0.450000,val0.450000，Lr1.000000e-03
Finishedep0ch17/200:cost1.642356,train:0.50000,val0.500000，1r1.000000e-03
Finishedep0ch18/200:c0st1.535239，train:0.600000,val0.600000,1r1.000000e-03
Finished ep0ch19/200:c0st1.421527，train:0.600000,val0.600000，Lr1.000000e-03
L A 20A BAC A AA AAAA AAAAAN A
1
Finished epoch 195/200:c0st0.002694,train:1.000000,val1.000000 r1.000000e-03
Finishedep0ch196/200:c05t0.002674，train:1.000000，val1.00000，Lr1.000000e-03
Finishedep0ch197/200:c0st0.002655,train:1.000000，val1.0000,lr1.000000e-03
Finished ep0ch198/200:c05t0.002635,train:1.00000，val1.000000，lr1.000000e-03
Finishedep0ch 199/200:c05t0.002617，train:1.000000，val1.00000，Lr1.000000e-03
Finished epoch200/200:c0st0.002597,train:1.00000，val1.000000,lr1.000000e-03
finished optimization.best validation accuracy:1.000000
```

Lets try to train now...

Start with small regularization and find learning rate that makes the loss go down.

```python
model=init_two_layer_model(32*32*3,50,10） #input size,hidden size,number of classes
trainer=ClassifierTrainer()
best_model,stats=trainer.train(X_train,y_train,X_val,y_val,
model,two_layer_net,
num_epochs=10，reg=0.000001,
update='sgd',learning_rate_decay=1,
sample_batches=True,
learning_rate=le-6,verbose=True)
```

## Lets try to train now...

Start with small regularization and find learning rate that makes the loss go down.

<!-- image-->

Notice train/val accuracy goes to 20%

## Now let's try learning rate 1e6

## Lets try to train now...

Start with small regularization and find learning rate that makes the loss go down.

loss not going down: learning rate too low loss exploding: learning rate too high cost: NaN almost always means high learning rate...

```csv
model=init_two_layer model(32*32*3,50，10） #input size,hidden size，number of classes
trainer=ClassifierTrainer()
best_model，stats=trainer.train(x_train，y_train，X_val,y_val，
model,two_layer_net,
num_epochs=10,reg=0.000001,
update='sgd',learning_rate_decay=1,
sample_batches =True,
learning_rate=le6,verbose=True)
/home/karpathy/cs23in/code/cs23in/classifiers/neural_net.py:50:Runtimewarning:divide by zero en
countered in log
data_loss=-np.sum（np.log(probsrange（N），yl））/N
/home/karpathy/cs23in/code/cs23in/classifiers/neural_net.py:48:Runtimewarning:invalid value enc
ountered in subtract
probs=np.exp(scores-np.max(scores,axis=1,keepdims=True))
Finished epoch1/10:cost nan,train:0.09100o,val 0.0870e0,lr1.000000e+06
Finished epoch2/10:cost nan,train:0.095000,val0.087000,lr1.000000e+06
Finished epoch3/10:cost nan，train:0.10oo0,val0.0870co,lr1.000000e+06
```

Lets try to train now...

Start with small regularization and find learning rate that makes the loss go down.

loss not going down: learning rate too low loss exploding: learning rate too high 3e-3 is still too high. Cost explodes...

```csv
model=init_two_layer_model(32*32*,50，10） #input size,hidden size,number of classes
trainer=ClassifierTrainer()
best_model，stats=trainer.train(X_train,y_train,X_val,y_val,
model,two_layer_net,
num_epochs=10，reg=0.000001,
update='sgd',learning_rate_decay=1,
sample_batches=True,
learning_rate=3e-3,verbose=True)
Finished epoch1/10:cost 2.186654,train:0.308000，val0.306000，Lr3.000000e-03
Finished ep0ch2/10:cost2.176230，train:0.330000,val0.350000,1r3.00000e-03
Finished ep0ch3/10:cost1.942257,train:0.376000,val0.352000,lr3.000000e-03
Finished ep0ch4/10:cost1.827868,train:0.329000,val0.310000,Lr3.000000e-03
Finished epoch5/10:costinf,train:0.128000,val0.128000，1r3.00o00e-03
Finishedepoch6/10:costinf，train:0.144000,val0.147000,lr3.000000e-03
```

=> Rough range for learning rate we should be cross-validating is somewhere [1e-3 ... 1e-5]

## How to do Cross Validation?

## coarse -> fine cross-validation in stages

First stage: only a few epochs to get rough idea of what params work   
Second stage: longer running time, finer search   
... (repeat as necessary)

Tip for detecting explosions in the solver:

If the cost is ever > 3 \* original cost, break out early

## For example: run coarse search for 5 epochs

## note it's best to optimize in log space! （ ■

trainer=ClassifierTrainer()   
model=init_two layermodel(32\*32\*3,50,10） #input size，hiddensize,number of classes   
trainer=ClassifierTrainer()   
best_model_local,stats=trainer.train(X_train,y_train,X_val,y_val,   
model,two_layer_net,   
numepochs=5,reg=reg,   
update='momentum',learning_rate_decay=0.9,   
sample_batches= True,batch_size=100,   
learning rate=lr,verbose=False)

val_acc:0.412000，lr:1.405206e-04，reg:4.793564e-01，(1/100)  
val_acc:0.214000,lr:7.231888e-06,reg:2.321281e-04,(2/ 100)  
val_acc:0.208000,lr:2.119571e-06,reg:8.011857e+01，（3/100)  
val_acc:0.196000, r:1.551131e-05，reg:4.374936e-05，（4/100)  
val_acc:0.079000，1r:1.753300e-05,reg:1.200424e+03,（5/100)  
valacc:0.223000，lr:4.215128e-05，reg:4.196174e+01，(6/100）  
val_acc:0.441000，lr:1.750259e-04，reg:2.110807e-04，（7/100）  
valacc:0.241000，lr:6.749231e-05，reg:4.226413e+01，（8/100)  
val_acc:0.482000，lr:4.296863e-04，reg:6.642555e-01，(9/100）  
val_acc:0.079000, r:5.401602e-06，reg:1.599828e+04， (10 100)  
val_acc:0.154000,lr:1.618508e-06,reg:4.925252e-01，（11/100）

## Now run finer search...

```python
max_count =100
forcount in xrange(max_count):
reg= 10**uniform（-5,5)
lr=10**uniform（-3,-6)
```

adjust range

max count =100   
forcount in xrange(max_count）：   
reg=10\*\*uniform(-4,0)   
r=10\*\*uniform（-3,-4)

```csv
val_acc:0.527000，1r:5.340517e-04，reg:4.097824e-01，（0/100)
val_acc ： 0.492000, t： Z 279484e -04 reg: 9.991345e-04, 100)
val_acc:0.512000, lr:8.680827e-04， reg:1.349727e-02， （2 /100)
val_acc:0.461000， lr:1.028377e-04， reg:1.220193e-02, （3/100)
val_acc:0.460000, lr:1.113730e-04， reg:5.244309e-02, （4/100）
val_acc:0.498000, Lr:9.477776e-04， reg:2.001293e-03, （5/100）
valacc:0.469000， lr:1.484369e-04， req:4.328313e-01, (6/100)
val_acc:0.522000，1r:5.586261e-04，reg:2.312685e-04，(7/100)
val acc: 0.530000. lr:5.808183e-04. req: 8.259964e-02. (8 / 100)
val_acc:0.489000, lr:1.979168e-04, reg:1.010889e-04,(9/100)
val_acc:0.490000, lr:2.036031e-04， reg:2.406271e-03，（10 / 100)
val_acc:0.475000， lr:2.021162e-04， reg:2.287807e-01,(11/100)
val_acc:0.460000, lr:1.135527e-04，reg:3.905040e-02,（12/100)
val acc:0.515000 lr:6.947668e-04, reg:1.562808e-02， (13/100)
valacc:0.531000，1r:9.471549e-04，reg:1.433895e-03，(14/100)
val_acc:0.509000, lr:3.140888e-04,reg:2.857518e-01, （15/ 100)
val_acc:0.514000, r:6.438349e-04， reg:3.033781e-01,(16/100）
val_acc:0.502000, r:3.921784e-04, reg:2.707126e-04,（17 / 100)
val_acc:0.509000, lr:9.752279e-04， reg:2.850865e-03, （18/ 100)
val_acc:0.500000 lr:2.412048e-04， reg:4.997821e-04,（19 / 100)
val_acc:0.466000, lr:1.319314e-04，reg:1.189915e-02， （20 100)
val_acc:0.516000，lr:8.039527e-04，reg:1.528291e-02,（21/100)
```

53% - relatively good for a 2-layer neural net with 50 hidden neurons.