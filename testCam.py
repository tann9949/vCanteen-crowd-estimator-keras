import cv2
import time
import datetime

def run():
    cv2.namedWindow("Test Cam")
    img_counter = 0

    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cv2.imshow("Test Cam", frame)
    t_end = round(int(time.time() + 300))
    while cam.isOpened():
        ret, frame = cam.read()
        cv2.imshow("Test Cam", frame)

        key = cv2.waitKey(1)
        
        if round(int(time.time())) == t_end:
            current_dt = datetime.datetime.now().strftime('%Y-%m-%d %H%M%S')
            img_name = str(current_dt)+'.png'
            cv2.imwrite(img_name, frame)
            print("{} is written!".format(img_name))
            t_end = round(int(time.time() + 300))
            
        if key & 0xFF == ord('q'):
            print("Closing the window")
            break

        elif key%256 == 32:
            # SPACE pressed
            img_name = "test_angle/VCanteen_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            current_dt = datetime.datetime.now()
            print(current_dt.strftime('%Y-%m-%d %H:%M:%S'))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()

run()