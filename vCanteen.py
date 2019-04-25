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



MAX_COUNT = 240
# URL = 'https://en04r5not39z8i.x.pipedream.net/'
URL = 'https://vcanteen.herokuapp.com/v2/crowd-estimation/prediction'

def send_JSON(pred, time):
    data = {
        'created_at': time,
        'percent_density': pred
    }
    r = requests.post(URL, json=data)
    return r.status_code

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


def run(model, videopath = 0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    output_text = ''
    sec = 5
    img_counter = 0
    text_color = (0,255,0)
    current_dt = datetime.datetime.now()
    
    cam = cv2.VideoCapture(videopath)
    ret, frame = cam.read()
    # cv2.imshow("Camera", frame)
    gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
    gray = (gray - 127.5) / 128
    inputs = np.reshape(gray, [1, gray.shape[0], gray.shape[1], 1])
    pred = round(np.sum(model.predict(inputs)))
    curr_time = current_dt.strftime('%Y-%m-%d %H:%M:%S')
    percent_den = int(pred*100/MAX_COUNT)
    output_text = str(curr_time)+ ' >> PRED : '+str(percent_den)+' %'
    print(output_text)
    send_JSON(percent_den, curr_time)
    # cv2.rectangle(frame, (10,10), (800, 20),(0,0,0),-1)
    # cv2.putText(frame, output_text, (10,30), font, 0.5, text_color, 1, cv2.LINE_AA)
    # cv2.imshow("Camera", frame)

    t_end = round(int(time.time() + sec))
    while cam.isOpened():
        ret, frame = cam.read()
        gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
        gray = (gray - 127.5) / 128
        # cv2.rectangle(frame, (10,15), (400, 35),(0,0,0),-1)
        # cv2.putText(frame, output_text, (10,30), font, 0.5, text_color, 1, cv2.LINE_AA)
        # cv2.imshow("Camera", frame)

        key = cv2.waitKey(1)
        
        if round(int(time.time())) == t_end:
            current_dt = datetime.datetime.now()
            inputs = np.reshape(gray, [1, gray.shape[0], gray.shape[1], 1])
            pred = round(np.sum(model.predict(inputs)))
            curr_time = current_dt.strftime('%Y-%m-%d %H:%M:%S')
            percent_den = int(pred*100/MAX_COUNT)
            output_text = str(curr_time)+ ' >> PRED : '+str(percent_den)+' %'
            print(output_text)
            send_JSON(percent_den, curr_time)
            # cv2.rectangle(frame, (10,15), (400, 35),(0,0,0),-1)
            # cv2.putText(frame, output_text, (10,30), font, 0.5, text_color, 1, cv2.LINE_AA)
            # cv2.imshow("Camera", frame)
            t_end = round(int(time.time() + sec))
        
        if key & 0xFF == ord('q'):
            print("Closing the window")
            break
        
        elif key % 256 == 32:
            # SPACE pressed
            inputs = np.reshape(gray, [1, gray.shape[0], gray.shape[1], 1])
            pred = round(np.sum(model.predict(inputs)))
            current_dt = datetime.datetime.now()
            percent_den = int(pred*100/MAX_COUNT)
            curr_time = current_dt.strftime('%Y-%m-%d %H:%M:%S')
            output_text = str(curr_time)+ ' >> PRED : '+str(percent_den)+' %'
            print(output_text)
            send_JSON(percent_den, curr_time)
            # cv2.rectangle(frame, (10,15), (400, 35),(0,0,0),-1)
            # cv2.putText(frame, output_text, (10,30), font, 0.5, text_color, 1, cv2.LINE_AA)
            # cv2.imshow("Camera", frame)


    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = get_MCNN()
    model.load_weights('keras_weight/trained_v2.h5')
    videopath = 'icanteen_vid/TEST_3.mp4'
    run(model, videopath)




