import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

def msqrt(pred, gt):
    res = 0
    num_samples = pred.shape[0]
    print(pred.T)
    print(gt)
    for j in range(num_samples):
        res = res + (pred[j]-gt[j])**2
    
    return (1.0/num_samples*res)	

# load data from npz-file
arr = np.load('data.npz')
data = arr['arr_0']

# let's shuffle 
np.random.shuffle(data)

# lets keep the first 10 elements for final testing
test_set  = data[0:10,0:2]
train_set = data[10:,0:2] 
size_train_set = train_set.shape[0]

X=train_set[:,0].reshape((size_train_set,1))
y=train_set[:,1].reshape((size_train_set,1))

# now create final train and validation set
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=42)
min_err      = 10000
best_lambda  = 0

for i in np.arange(0.1,20,1):
    reg = Lasso(alpha=i)

    # X_train has to be replaced by matrix (1 x x**2, x**3, ...
    reg.fit(X_train, y_train)
    
    # check regressor on validation data set
    res = reg.predict(X_val)
    err = msqrt(res, y_val)
    
    if (err < min_err):
        min_err = err
        best_lambda = i

print('min error = {}'.format(min_err))		
print('best value of lambda = {}'.format(best_lambda))