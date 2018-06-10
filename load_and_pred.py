import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential,save_model,load_model




img_rows, img_cols = 28, 28
num_classes = 10


def prep_data(raw, train_size, val_size):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y





fashion_model = load_model("/home/aurash/Fashion_kaggle/model.hdf5",custom_objects = None, compile = False)


fashion_test = "fashion-mnist_test.csv"
fashion_data = np.loadtxt(fashion_test, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data, train_size=50000, val_size=5000)

pred = fashion_model.predict(x)

county=0
for xe in pred:
	maxval = 0
	count = -1
	counter = 0 
	for val in xe:
		count+=1
		if val > maxval:
			maxval = val
			counter = count
	print(str(counter)+" "+ str(y[county]))
	county+=1