import numpy as np
import cv2
import sys
import json
import time
import os
#import copy
from keras.models import model_from_json
emo_labels = ['angry', 'disgust:', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_class = len(emo_labels)
root_path = './pic/'
img_path = './fer2013/'

def predict_emotion(face_img):
    face_img=face_img*(1./255)
    resized_img=cv2.resize(face_img,(img_size,img_size))#,interpolation=cv2.INTER_LINEAR
    rsh_image=resized_img.reshape(1,img_size,img_size,1)
    result = model.predict_classes(rsh_image,batch_size=32,verbose=0)#predict
    return result
def predict_emotion_mul(face_img):
    face_img=face_img*(1./255)
    resized_img=cv2.resize(face_img,(img_size,img_size))#,interpolation=cv2.INTER_LINEAR
    rsz_img=[]
    rsh_img=[]
    results=[]
    #print (len(resized_img[0]),type(resized_img))
    rsz_img.append(resized_img[:,:])#resized_img[1:46,1:46]
    rsz_img.append(resized_img[2:45,:])
    rsz_img.append(cv2.flip(rsz_img[0],1))
    #rsz_img.append(cv2.flip(rsz_img[1],1))

    
    #rsz_img.append(resized_img[:,:])
    i=0
    for rsz_image in rsz_img:
        rsz_img[i]=cv2.resize(rsz_image,(img_size,img_size))
        #=========================
        '''cv2.HoughLinesP
        cv2.namedWindow('%d'%i,0)
        cv2.resizeWindow('%d'%i,(250,250))'''
        #=========================
        #cv2.imshow('%d'%i,rsz_img[i])
        i+=1
    for rsz_image in rsz_img:
        rsh_img.append(rsz_image.reshape(1,img_size,img_size,1))
    i=0
    for rsh_image in rsh_img:
        list_of_list = model.predict_proba(rsh_image,batch_size=32,verbose=0)#predict
        result = [prob for lst in list_of_list for prob in lst]
        results.append(result)
    result_sum=np.array([0]*num_class)
    for result in results:
        result_sum=result_sum+np.array(result)
    return np.argmax(result_sum)
def prodece_confusion_matrix(Images,total_num):
    results=np.array([0]*num_class)
    total=[]
    for images in Images:
        #print(len(images))
        for image in images:
            #print (image)
            img=cv2.imread(image)
            img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            results[predict_emotion_mul(img_gray)]+=1
        print(results,results/len(images))
        total.append(results)
        results=np.array([0]*num_class)
    sum=0
    for i in range(len(total[0])):
        sum+=total[i][i]
    print('acc:',sum*100./total_num,'%')
    print(model_path)
if __name__ == '__main__':
    images=[]
    Images=[]
    total_num=0
    if len(sys.argv)==2:

        #model_path=root_path+'model5_70epoch
        model_path=root_path+sys.argv[1]+'/'
        img_size=48
        emotion_labels = emo_labels

        #load json and create model arch
        json_file=open(model_path+'model_json.json')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        #load weight
        model.load_weights(model_path+'model_weight.h5')
        dir=img_path+'test'#sys.argv[1]
        if os.path.isdir(dir):
            files=os.listdir(dir)
            for file in files:
                if os.path.isdir(dir+'/'+file):
                    #print('in path')
                    imgs=os.listdir(dir+'/'+file)
                    #print(len(imgs))
                    for img in imgs:
                        if img.endswith('jpg') or img.endswith('png'):
                            images.append(dir+'/'+file+'/'+img)
                    total_num+=len(images)
                    Images.append(images)
                    images=[]
        prodece_confusion_matrix(Images,total_num)
    else:
        print('there should be a parameter after .py')
    

