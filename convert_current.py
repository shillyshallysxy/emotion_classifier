import os
import cv2
import sys
pad='a'
root_path='./fer2013/test/'
rootPath='./'
cascPath=rootPath+'haarcascade_frontalface_alt.xml'
faceCasccade=cv2.CascadeClassifier(cascPath)
num=0
if __name__=='__main__':
    if len(sys.argv)==2:
        path=root_path+sys.argv[1]
        path_w=root_path+pad+sys.argv[1]
        #print(path,path_w)
        if os.path.isdir(path):
            if not os.path.exists(path_w):
                os.makedirs(path_w)
            lists=os.listdir(path)
            for list in lists:
                #print(list)
                img=cv2.imread(path+'/'+list)
                img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces = faceCasccade.detectMultiScale(
                        img_gray,
                        scaleFactor=1.1,
                        minNeighbors=1,
                        minSize=(35,35),
                        )
                for x,y,w,h in faces:
                    face_img_gray = img_gray[y:y+h,x:x+w]
                    print (path_w+'/%d'%num+'.jpg')
                    cv2.imwrite(path_w+'/%d'%num+'.jpg',face_img_gray)
                    num+=1