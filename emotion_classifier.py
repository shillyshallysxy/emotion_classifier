import numpy as np
import cv2
import sys
import json
import time
import os
# import copy
from keras.models import model_from_json

root_path = './pic/'
model_path = root_path + '/model/'  # '/model_0.7/'
img_size = 48
emotion_labels = ['angry', 'disgust:', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_class = len(emotion_labels)
# load json and create model arch
json_file = open(model_path + 'model_json.json')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weight
model.load_weights(model_path + 'model_weight.h5')


def predict_emotion(face_img):
    face_img = face_img * (1. / 255)
    resized_img = cv2.resize(face_img, (img_size, img_size))  # ,interpolation=cv2.INTER_LINEAR
    rsz_img = []
    rsh_img = []
    results = []
    # print (len(resized_img[0]),type(resized_img))
    rsz_img.append(resized_img[:, :])  # resized_img[1:46,1:46]
    rsz_img.append(resized_img[2:45, :])
    rsz_img.append(cv2.flip(rsz_img[0], 1))
    # rsz_img.append(cv2.flip(rsz_img[1],1))

    '''rsz_img.append(resized_img[0:45,0:45])
    rsz_img.append(resized_img[2:47,0:45])
    rsz_img.append(resized_img[2:47,2:47])
    rsz_img.append(cv2.flip(rsz_img[2],1))
    rsz_img.append(cv2.flip(rsz_img[3],1))
    rsz_img.append(cv2.flip(rsz_img[4],1))'''
    i = 0
    for rsz_image in rsz_img:
        rsz_img[i] = cv2.resize(rsz_image, (img_size, img_size))
        # =========================
        # cv2.imshow('%d'%i,rsz_img[i])
        i += 1
    # why 4 parameters here, what's it means?
    for rsz_image in rsz_img:
        rsh_img.append(rsz_image.reshape(1, img_size, img_size, 1))
    i = 0
    for rsh_image in rsh_img:
        list_of_list = model.predict_proba(rsh_image, batch_size=32, verbose=1)  # predict
        result = [prob for lst in list_of_list for prob in lst]
        results.append(result)
    return results


def face_detect(image_path):
    # rootPath='E:\Python\Opencv\opencv\sources\data\haarcascades\\'
    cascPath = './haarcascade_frontalface_alt.xml'
    faceCasccade = cv2.CascadeClassifier(cascPath)

    # load the img and convert it to bgrgray
    # img_path=image_path
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # face detection
    faces = faceCasccade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(30, 30),
    )
    # print('img_gray:',type(img_gray))
    return faces, img_gray, img


if __name__ == '__main__':
    images = []
    flag = 0
    if len(sys.argv) != 1:
        dir = root_path + sys.argv[1]
        if os.path.isdir(dir):
            files = os.listdir(dir)
            print(files)
            for file in files:
                if file.endswith('jpg') or file.endswith('png') or file.endswith('PNG') or file.endswith('JPG'):
                    images.append(dir + '/' + file)
        else:
            images.append(dir)
    else:
        print('there should be a parameter after .py')
    if len(sys.argv) == 3:
        flag = 1

    for image in images:
        # print (image)
        faces, img_gray, img = face_detect(image)
        spb = img.shape
        sp = img_gray.shape
        height = sp[0]
        width = sp[1]
        size = 600
        if flag == 0:
            emo = ""
            face_exists = 0
            for (x, y, w, h) in faces:
                face_exists = 1
                face_img_gray = img_gray[y:y + h, x:x + w]
                results = predict_emotion(face_img_gray)  # face_img_gray
                result_sum = np.array([0]*num_class)
                for result in results:
                    result_sum = result_sum + np.array(result)
                    print(result)
                angry, disgust, fear, happy, sad, surprise, neutral = result_sum
                # 输出所有情绪的概率
                print(result_sum)
                print('angry:', angry, 'disgust:', disgust, ' fear:', fear, ' happy:', happy, ' sad:', sad,
                      ' surprise:', surprise, ' neutral:', neutral)
                label = np.argmax(result_sum)
                emo = emotion_labels[label]
                print('Emotion : ', emo)
                # 输出最大概率的情绪
                t_size = 2
                ww = int(spb[0] * t_size / 300)
                www = int((w + 10) * t_size / 100)
                www_s = int((w + 20) * t_size / 100) * 2 / 5
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), ww)
                cv2.putText(img, emo, (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            www_s, (255, 0, 255), thickness=www, lineType=1)
                # img_gray full face     face_img_gray part of face
            if face_exists:
                cv2.HoughLinesP
                cv2.namedWindow(emo, 0)
                cent = int((height * 1.0 / width) * size)
                cv2.resizeWindow(emo, (size, cent))
                cv2.imshow(emo, img)
                k = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if k & 0xFF == ord('q'):
                    break
        elif flag == 1:
            img = cv2.imread(image)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            results = predict_emotion(img_gray)  # face_img_gray
            result_sum = np.array([0]*num_class)
            for result in results:
                result_sum = result_sum + np.array(result)
                print(result)
            angry, disgust, fear, happy, sad, surprise, neutral = result_sum
            # 输出所有情绪的概率
            print(result_sum)
            print('angry:', angry, 'disgust:', disgust, ' fear:', fear, ' happy:', happy, ' sad:', sad,
                  ' surprise:', surprise, ' neutral:', neutral)
            label = np.argmax(result_sum)
            emo = emotion_labels[label]
            print('Emotion : ', emo)
            # 输出最大概率的情绪

            # img_gray full face     face_img_gray part of face
            cv2.HoughLinesP
            # cv2.imwrite('./'+emo+'.jpg',face_img_gray)
            cv2.namedWindow(emo, 0)
            size = 400
            cent = int((height * 1.0 / width) * size)
            cv2.resizeWindow(emo, (size, cent))

            cv2.imshow(emo, img)
            cv2.waitKey(0)

