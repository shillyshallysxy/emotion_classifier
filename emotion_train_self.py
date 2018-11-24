import keras
from keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
batch_siz = 128
num_classes = 7
nb_epoch = 70
img_size=48
root_path='./fer2013'
class Model:
    def __init__(self):
        self.model = None

    def build_model(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, (1, 1), strides=1, padding='same', input_shape=(img_size, img_size, 1)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(2048))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))
        self.model.summary()
    def train_model(self):
        sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                #optimizer='rmsprop',
                metrics=['accuracy'])
        #自动扩充训练样本
        train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip=True)
        #归一化验证集
        val_datagen = ImageDataGenerator(
                rescale = 1./255)
        eval_datagen = ImageDataGenerator(
                rescale = 1./255)
        #以文件分类名划分label
        train_generator = train_datagen.flow_from_directory(
                root_path+'/train',
                target_size=(img_size,img_size),
                color_mode='grayscale',
                batch_size=batch_siz,
                class_mode='categorical')
        val_generator = val_datagen.flow_from_directory(
                root_path+'/val',
                target_size=(img_size,img_size),
                color_mode='grayscale',
                batch_size=batch_siz,
                class_mode='categorical')
        eval_generator = eval_datagen.flow_from_directory(
                root_path+'/test',
                target_size=(img_size,img_size),
                color_mode='grayscale',
                batch_size=batch_siz,
                class_mode='categorical')
        early_stopping = EarlyStopping(monitor='loss',patience=3)
        history_fit=self.model.fit_generator(
                train_generator,
                steps_per_epoch=800/(batch_siz/32),#28709
                nb_epoch=nb_epoch,
                validation_data=val_generator,
                validation_steps=2000,
                #callbacks=[early_stopping]
                )
#         history_eval=self.model.evaluate_generator(
#                 eval_generator,
#                 steps=2000)
        history_predict=self.model.predict_generator(
                eval_generator,
                steps=2000)
        with open(root_path+'/model_fit_log','w') as f:
            f.write(str(history_fit.history))
        with open(root_path+'/model_predict_log','w') as f:
            f.write(str(history_predict))
#         print("%s: %.2f%%" % (self.model.metrics_names[1], history_eval[1] * 100))
        print('model trained')
    def save_model(self):
        model_json=self.model.to_json()
        with open(root_path+"/model_json.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(root_path+'/model_weight.h5')
        self.model.save(root_path+'/model.h5')
        print('model saved')

if __name__=='__main__':
    model=Model()
    model.build_model()
    print('model built')
    model.train_model()
    print('model trained')
    model.save_model()
    print('model saved')
