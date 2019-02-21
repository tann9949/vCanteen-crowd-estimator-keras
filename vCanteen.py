from keras.layers import Dense, Input, Conv2D, MaxPooling2D, concatenate
from keras.models import Model
from keras.optimizers import Adam
import cv2
import time
import datetime
import keras
import numpy as np
import os
import math
from keras import models
import requests

def send_JSON(pred, time):
    db_path = 'https://enidwk62vn7d.x.pipedream.net/'
    data = {
        'time_stamp': time,
        'crowd_size': pred
    }
    
    r = requests.post(db_path, data=data)

def get_MCNN():
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
    model.compile(optimizer=Adam(0.001),
                  loss='mean_absolute_error',
                  metrics=['mean_absolute_error'])
                  
    return model

def run(model):
    font = cv2.FONT_HERSHEY_SIMPLEX
    output_text = ''
    sec = 5
    cv2.namedWindow("Camera")
    img_counter = 0
    
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cv2.imshow("Camera", frame)
    t_end = round(int(time.time() + sec))
    while cam.isOpened():
        ret, frame = cam.read()
        gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
#        cv2.imshow("Camera", frame)
        cv2.putText(frame, output_text, (10,30), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1)
        
        if round(int(time.time())) == t_end:
            current_dt = datetime.datetime.now()
#            img_name = str(current_dt)+'.png'
#            cv2.imwrite(img_name, frame)
            inputs = np.reshape(gray, [1, gray.shape[0], gray.shape[1], 1])
            pred = round(np.sum(model.predict(inputs)))
#            print("{} is written!".format(img_name))
            curr_time = current_dt.strftime('%Y-%m-%d %H:%M:%S')
            output_text = str(curr_time)+ ' >> PRED : '+str(pred)+' people'
            print(output_text)
            send_JSON(pred, curr_time)
            cv2.putText(frame, output_text, (10,30), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
            cv2.imshow("Camera", frame)
            t_end = round(int(time.time() + sec))
        
        if key & 0xFF == ord('q'):
            print("Closing the window")
            break
        
        elif key % 256 == 32:
            # SPACE pressed
#            img_name = "VCanteen_{}.png".format(img_counter)
#            cv2.imwrite(img_name, frame)
            inputs = np.reshape(gray, [1, gray.shape[0], gray.shape[1], 1])
            pred = round(np.sum(model.predict(inputs)))
#            print("{} written!".format(img_name))
            current_dt = datetime.datetime.now()
            curr_time = current_dt.strftime('%Y-%m-%d %H:%M:%S')
            output_text = str(curr_time)+ ' >> PRED : '+str(pred)+' people'
            print(output_text)
            send_JSON(pred, curr_time)
            cv2.putText(frame, output_text, (10,30), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
            cv2.imshow("Camera", frame)
            img_counter += 1


    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = get_MCNN()
    model.load_weights('keras_weight/weights.h5')

    for layer in model.layers[:-2]:
        layer.trainable = False

    mcnn = models.Sequential()
    mcnn.add(model)
    mcnn.load_weights('keras_weight/trained.h5')
    run(mcnn)




