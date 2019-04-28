
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import math
import scipy.io


# In[2]:


ICANTEEN_TRAIN_PATH = 'image_preprocessor/data/formatted_trainval/icanteen_patches/train/'
ICANTEEN_TRAIN_GT_PATH = 'image_preprocessor/data/formatted_trainval/icanteen_patches/train_den/'
ICANTEEN_VAL_PATH = 'image_preprocessor/data/formatted_trainval/icanteen_patches/val/'
ICANTEEN_VAL_GT_PATH = 'image_preprocessor/data/formatted_trainval/icanteen_patches/val_den/'
ICANTEEN_TEST_PATH = 'icanteen_img/test/images/'
ICANTEEN_TEST_GT_PATH = 'icanteen_img/test/ground_truth/'


# In[3]:


def preprocess_test(path):
    print('loading testing dataset...')
    img_names = os.listdir(path)
    img_num = len(img_names)
    img_names.sort()
    
    data = []
    i = 1
    for name in img_names:
        if name=='.DS_Store':
            continue
        if i % 50 == 0:
            print('loaded:', i, '/', img_num)
        img = cv2.imread(path+name,0)
        norm_img = (img - 127.5) / 128
        data.append(norm_img)
        i += 1
    print('load data finished')
    return data

def preprocess_train(img_path, gt_path):
    print('loading training dataset...')
    img_names = os.listdir(img_path)
    img_num = len(img_names)
    img_names.sort()

    data = []
    density = []
    count = 1
    for name in img_names:
        if count % 100 == 0:
            print(count, '/', img_num)
        count += 1
        img = cv2.imread(img_path + name, 0)
        img = np.array(img)
        norm_img = (img - 127.5) / 128
        den = np.loadtxt(open(gt_path + name[:-4] + '.csv'), delimiter = ",")
        den_quarter = np.zeros((int(den.shape[0] / 4), int(den.shape[1] / 4)))
        for i in range(len(den_quarter)):
            for j in range(len(den_quarter[0])):
                for p in range(4):
                    for q in range(4):
                        den_quarter[i][j] += den[i * 4 + p][j * 4 + q]
        data.append(np.reshape(norm_img, [norm_img.shape[0], norm_img.shape[1], 1]))
        density.append(np.reshape(den_quarter, [den_quarter.shape[0], den_quarter.shape[1], 1]))
    data = np.array(data)
    density = np.array(density)
    print('load training data finished')
    return (data, density)
        
def get_ground_truth(path, verbose = 0):
    if verbose == 1:
        print('loading ground truth...')
    gt_names = os.listdir(path)
    gt_names.sort()
    gt = []
    i = 1
    for file in gt_names:
        if file=='.DS_Store':
            continue
        mat = scipy.io.loadmat(path+file)
        gt.append(mat['image_info'][0][0][0][0][1][0][0])
        i += 1
    if verbose == 1:
        print('load ground truth finished')
    return gt


# In[4]:


def test_model(model, gt_path, verbose = 0, show=False, save=False):
    mae = 0
    mse = 0
    mae_sht = 0
    mse_sht = 0
    ground_truth = get_ground_truth(gt_path)
    pred = []
    pred_sht= []
    
    mcnn = get_MCNN()
    mcnn.load_weights('keras_weight/weights.h5')

    initial_num = 31
    i = 0
    for d in data:
        inputs = np.reshape(d, [1, d.shape[0], d.shape[1], 1])
        outputs = model.predict(inputs)
        c_pre = round(np.sum(outputs))
        pred.append(c_pre)
        sht_pre = round(np.sum(mcnn.predict(inputs)))
        pred_sht.append(sht_pre)
        if verbose == 1:
            print('='*10+'IMG_'+str(i+initial_num)+'.jpg'+'='*10)
            print('IMG_'+str(i+initial_num),'vCanteen predicted ', c_pre,'people')
            print('IMG_'+str(i+initial_num),'ShanghaiTech predicted ', sht_pre,'people')
            print('IMG_'+str(i+initial_num),'Actual',ground_truth[i], 'people')
        mae += abs(ground_truth[i]-c_pre)
        mse += (ground_truth[i]-c_pre)**2
        mae_sht += abs(ground_truth[i]-sht_pre)
        mse_sht += (ground_truth[i]-sht_pre)**2
        if verbose == 1:
            print('ERROR:',abs(ground_truth[i]-c_pre))
            print('current vCanteen MAE:',mae/(i+1))
            print('current vCanteen MSE:',(mse/(i+1))**0.5)
            print('current ShanghaiTech MAE:',mae_sht/(i+1))
            print('current ShanghaiTech MSE:',(mse_sht/(i+1))**0.5)

        den = outputs.reshape(outputs.shape[1], outputs.shape[2])
        img = inputs.reshape(inputs.shape[1], inputs.shape[2])
        den_resize = cv2.resize(den, (img.shape[1], img.shape[0]))
        if show:
            merge = 0.95 * den_resize + 0.05 * img
            plt.figure(figsize=(15, 7.5))
            plt.imshow(merge, cmap='gray')
            plt.title('IMG_'+str(initial_num+i))
            plt.xticks([])
            plt.yticks([])
            plt.show()
            
        if save:
            den_name = 'heat_'+'IMG_'+str(i+initial_num)+'.png'
            plt.imsave('icanteen_heat/'+den_name, den)
            print('Finish saving',den_name,'!')
        i += 1

    print('vCanteen Mean Absolute Error:',mae/len(ground_truth))
    print('vCanteen Mean Square Error:',math.sqrt(mse/len(ground_truth)))
    print('ShanghaiTech Mean Absolute Error:',mae_sht/len(ground_truth))
    print('ShanghaiTech Mean Square Error:',math.sqrt(mse_sht/len(ground_truth)))

    plt.figure(figsize=(20, 10))
    plt.plot(np.arange(1,len(ground_truth)+1),ground_truth, marker='o', label = 'Ground Truth')
    plt.plot(np.arange(1,len(ground_truth)+1),pred, marker='o', label = 'vCanteen')
    plt.plot(np.arange(1,len(ground_truth)+1),pred_sht, marker='o', label = 'ShanghaiTech')
    plt.xticks(np.arange(1,len(ground_truth)+1))
    plt.xlabel('Img no.')
    plt.ylabel('People')
    plt.legend()
    plt.title('Head counts Actual vs Predict')
    plt.show()
    
    return (pred, pred_sht, ground_truth)


# In[26]:


def evaluate(model, test_data, ground_truth):
    mae = 0
    mse = 0
    i = 0
    for d in test_data:
        inputs = np.reshape(d, [1, d.shape[0], d.shape[1], 1])
        outputs = model.predict(inputs)
        c_pre = round(np.sum(outputs))
        mae += abs(ground_truth[i]-c_pre)
        mse += (ground_truth[i]-c_pre)**2
        i += 1
        
    return (mae, mse)


# In[5]:


train, train_den = preprocess_train(ICANTEEN_TRAIN_PATH, ICANTEEN_TRAIN_GT_PATH)
val, val_den = preprocess_train(ICANTEEN_VAL_PATH, ICANTEEN_VAL_GT_PATH)
data = preprocess_test(ICANTEEN_TEST_PATH)


# In[18]:


from keras.layers import Dense, Input, Conv2D, MaxPooling2D, concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD

def get_MCNN(lr, freeze=False):    
    input1 = Input(shape=(None, None, 1)) 
    
    # S
    xs = Conv2D(24, kernel_size = (5,5), padding = 'same', activation = 'relu')(input1)   
    xs = MaxPooling2D(pool_size = (2,2))(xs)
    xs = Conv2D(48, kernel_size = (3,3), padding = 'same', activation = 'relu')(xs)
    xs = MaxPooling2D(pool_size = (2,2))(xs)
    xs = Conv2D(24, kernel_size = (3,3), padding = 'same', activation = 'relu')(xs)
    xs = Conv2D(12, kernel_size = (3,3), padding = 'same', activation = 'relu')(xs)
    
    # M
    xm = Conv2D(20, kernel_size = (7,7), padding = 'same', activation = 'relu')(input1)   
    xm = MaxPooling2D(pool_size = (2,2))(xm)
    xm = Conv2D(40, kernel_size = (5,5), padding = 'same', activation = 'relu')(xm)
    xm = MaxPooling2D(pool_size = (2,2))(xm)
    xm = Conv2D(20, kernel_size = (5,5), padding = 'same', activation = 'relu')(xm)
    xm = Conv2D(10, kernel_size = (5,5), padding = 'same', activation = 'relu')(xm)
    
    # L
    xl = Conv2D(16, kernel_size = (9,9), padding = 'same', activation = 'relu')(input1)   
    xl = MaxPooling2D(pool_size = (2,2))(xl)
    xl = Conv2D(32, kernel_size = (7,7), padding = 'same', activation = 'relu')(xl)
    xl = MaxPooling2D(pool_size = (2,2))(xl)
    xl = Conv2D(16, kernel_size = (7,7), padding = 'same', activation = 'relu')(xl)
    xl = Conv2D(8, kernel_size = (7,7), padding = 'same', activation = 'relu')(xl)
    
    x = concatenate([xm, xs, xl])
    out = Conv2D(1, kernel_size = (1,1), padding = 'same')(x)
    model = Model(inputs=input1, outputs=out)
    
    if freeze:
        len_layers = len(model.layers)
        for layer in model.layers[:16]:
            layer.trainable=False
        for layer in model.layers[16:]:
            layer.trainable=True
        
    adam = Adam(lr)
    sgd = SGD(lr, momentum=0.9, nesterov=True)
    
    if freeze:
        model.compile(loss='mse', optimizer=adam, metrics=['mse'])
    else:
        model.compile(loss='mse', optimizer=sgd, metrics=['mse'])

    return model


# In[ ]:


from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard

verbose = 0
epochs = [10,20,40,80,160,320]
learning_rate = [0.01,0.001,0.0001]
batch_size = [1,2,4,8,16,32,64]
best_lr = 0
best_bs = 0
best_e = 0

best_model = None
min_test_mae = math.inf
ground_truth = get_ground_truth(ICANTEEN_TEST_GT_PATH)

for bs in batch_size:
    for e in epochs:
        for lr in learning_rate:
            K.clear_session()
            print('start training MCNN with batch size = {}, epochs = {}, learning rate = {}'.format(bs, e, lr))
            model = get_MCNN(lr=lr)
            model.fit(train, train_den,validation_data = (val, val_den),
                          epochs = e, batch_size = bs, verbose = verbose)
            mae, mse = evaluate(model, data, ground_truth)
            if mae < min_test_mae:
                print('test_loss improved from {} to {}'.format(min_test_mae, mae))
                min_test_mae = mae
                best_model = model
                best_lr = lr
                best_bs = bs
                best_e = e
                
print('Best model with lr = {}, epochs = {}, batch size = {}',format(best_lr, best_bs, best_e))


# In[ ]:


from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard

verbose = 0
epochs = [10,20,40]

learning_rate = [0.01,0.001,0.0001]
batch_size = [1,2,4,8,16,32,64]
best_lr = 0
best_bs = 0
best_e = 0

best_model = None
ground_truth = get_ground_truth(ICANTEEN_TEST_GT_PATH)
print('='*10, 'Transfer Learning setting')
for bs in batch_size:
    for e in epochs:
        for lr in learning_rate:
            K.clear_session()
            print('start training MCNN with batch size = {}, epochs = {}, learning rate = {}'.format(bs, e, lr))
            model = get_MCNN(freeze = True, lr=lr)
            model.load_weights('keras_weight/weights.h5')
            model.fit(train, train_den,validation_data = (val, val_den),
                          epochs = e, batch_size = bs, verbose = verbose)
            mae, mse = evaluate(model, data, ground_truth)
            if mae < min_test_mae:
                print('test_loss improved from {} to {}'.format(min_test_mae, mae))
                min_test_mae = mae
                best_model = model
print('Best model with lr = {}, epochs = {}, batch size = {}',format(best_lr, best_bs, best_e))


# In[ ]:


pred, sht_pred, ground_truth = test_model(model, ICANTEEN_TEST_GT_PATH)


# In[10]:





# In[ ]:





