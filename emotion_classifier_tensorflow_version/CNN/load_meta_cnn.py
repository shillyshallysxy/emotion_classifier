import os
import tensorflow as tf
import cv2
import numpy as np
import csv
import tqdm

data_folder_name = '..\\temp'
data_path_name = 'cv'
pic_path_name = 'pic'
cv_path_name = 'fer2013'
csv_file_name = 'fer2013.csv'
ckpt_name = 'cnn_emotion_classifier.ckpt'
casc_name = 'haarcascade_frontalface_alt.xml'
cv_path = os.path.join(data_folder_name, data_path_name, cv_path_name)
csv_path = os.path.join(cv_path, csv_file_name)
ckpt_path = os.path.join(data_folder_name, data_path_name, ckpt_name)
casc_path = os.path.join(data_folder_name, data_path_name, casc_name)
pic_path = os.path.join(data_folder_name, data_path_name, pic_path_name)
eval_path = os.path.join(data_folder_name, data_path_name, 'fer2013', 'test')

img_size = 48
confusion_matrix = True
emotion_labels = ['angry', 'disgust:', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_class = len(emotion_labels)


# config=tf.ConfigProto(log_device_placement=True)
sess = tf.Session()

saver = tf.train.import_meta_graph(ckpt_path+'.meta')
saver.restore(sess, ckpt_path)
graph = tf.get_default_graph()
name = [n.name for n in graph.as_graph_def().node]
print(name)
x_input = graph.get_tensor_by_name('x_input:0')
dropout = graph.get_tensor_by_name('dropout:0')
logits = graph.get_tensor_by_name('project/output/logits:0')


def prodece_confusion_matrix(images_, total_num_):
    results = np.array([0]*num_class)
    total = []
    for imgs_ in images_:
        for img_ in imgs_:
            results[np.argmax(predict_emotion(img_))] += 1
        print(results, np.around(results/len(imgs_), decimals=3))
        total.append(results)
        results = np.array([0]*num_class)
    sum = 0
    for i_ in range(num_class):
        sum += total[i_][i_]
    print('acc: {:.3f} %'.format(sum*100./total_num_))
    print('Using ', ckpt_name)


def predict_emotion(face_img_, img_size_=48):
    face_img_ = face_img_ * (1. / 255)
    resized_img_ = cv2.resize(face_img_, (img_size_, img_size_))  # ,interpolation=cv2.INTER_LINEAR
    rsz_img = []
    rsz_img.append(resized_img_[:, :])
    rsz_img.append(resized_img_[2:45, :])
    rsz_img.append(cv2.flip(rsz_img[0], 1))
    for i_, rsz_image in enumerate(rsz_img):
        rsz_img[i_] = cv2.resize(rsz_image, (img_size_, img_size_)).reshape(img_size_, img_size_, 1)
    rsz_img = np.array(rsz_img)
    feed_dict_ = {x_input: rsz_img, dropout: 1.0}
    pred_logits_ = sess.run([tf.reduce_sum(tf.nn.softmax(logits), axis=0)], feed_dict_)
    return np.squeeze(pred_logits_)


def face_detect(image_path, casc_path_=casc_path):
    if os.path.isfile(casc_path_):
        face_casccade_ = cv2.CascadeClassifier(casc_path_)
        img_ = cv2.imread(image_path)
        img_gray_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        # face detection
        faces = face_casccade_.detectMultiScale(
            img_gray_,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(30, 30),
        )
        return faces, img_gray_, img_
    else:
        print("There is no {} in {}".format(casc_name, casc_path_))


if __name__ == '__main__':
    if not confusion_matrix:
        images_path = []
        files = os.listdir(pic_path)
        for file in files:
            if file.lower().endswith('jpg') or file.endswith('png'):
                images_path.append(os.path.join(pic_path, file))
        for image in images_path:
            faces, img_gray, img = face_detect(image)
            spb = img.shape
            sp = img_gray.shape
            height = sp[0]
            width = sp[1]
            size = 600
            emotion_pre_dict = {}
            face_exists = 0
            for (x, y, w, h) in faces:
                face_exists = 1
                face_img_gray = img_gray[y:y + h, x:x + w]
                results_sum = predict_emotion(face_img_gray)  # face_img_gray
                for i, emotion_pre in enumerate(results_sum):
                    emotion_pre_dict[emotion_labels[i]] = emotion_pre
                # 输出所有情绪的概率
                print(emotion_pre_dict)
                label = np.argmax(results_sum)
                emo = emotion_labels[int(label)]
                print('Emotion : ', emo)
                # 输出最大概率的情绪
                # 使框的大小适应各种像素的照片
                t_size = 2
                ww = int(spb[0] * t_size / 300)
                www = int((w + 10) * t_size / 100)
                www_s = int((w + 20) * t_size / 100) * 2 / 5
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), ww)
                cv2.putText(img, emo, (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            www_s, (255, 0, 255), thickness=www, lineType=1)
                # img_gray full face     face_img_gray part of face
            if face_exists:
                cv2.namedWindow('Emotion_classifier', 0)
                cent = int((height * 1.0 / width) * size)
                cv2.resizeWindow('Emotion_classifier', size, cent)
                cv2.imshow('Emotion_classifier', img)
                k = cv2.waitKey(0)
                cv2.destroyAllWindows()
                # if k & 0xFF == ord('q'):
                #     break
    if confusion_matrix:
        with open(csv_path, 'r') as f:
            csvr = csv.reader(f)
            header = next(csvr)
            rows = [row for row in csvr]
            val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
            # tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
        confusion_images_total = []
        confusion_images = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        test_set = val
        total_num = len(test_set)
        for label_image_ in test_set:
            label_ = int(label_image_[0])
            image_ = np.reshape(np.asarray([int(p) for p in label_image_[-1].split()]), [img_size, img_size, 1])
            confusion_images[label_].append(image_)
        prodece_confusion_matrix(confusion_images.values(), total_num)
