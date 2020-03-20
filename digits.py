from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import Adam

from keras.layers import Input, Flatten, Dense
from keras.models import Model


(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

NUM_CLASSES = len(set(yTrain.T))

xTrain = xTrain / 255.0
xTest = xTest / 255.0

yTrain = to_categorical(yTrain, NUM_CLASSES)
yTest = to_categorical(yTest, NUM_CLASSES)

input_layer = Input(shape=(28, 28))

x = Flatten()(input_layer)

x = Dense(units=200, activation='relu')(x)
x = Dense(units=100, activation='relu')(x)
x = Dense(units=400, activation='relu')(x)
x = Dense(units=100, activation='relu')(x)

output_layer = Dense(units=10, activation='softmax')(x)

model = Model(input_layer, output_layer)

optimizer = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(xTrain, yTrain, epochs=10)

print(model.evaluate(xTest, yTest))

model.save('./digits_model.h5')