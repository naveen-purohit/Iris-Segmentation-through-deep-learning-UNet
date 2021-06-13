import UNet_iris_segmentation
from keras import backend as K
from keras.preprocessing.image import array_to_img
import cv2
import glob
import numpy as np
import os

if __name__ == "__main__":
    path= os.getcwd()
    data_path = path+"/"+"test"
    
    model_path = path
    print(model_path)
    print(data_path)

    img_type = "bmp"

    imgs = glob.glob(data_path + "/*." + img_type)
    print(imgs)
    # import the model
    model = UNet_iris_segmentation.create_model()

    # load the model
    model.load_weights(model_path + '/unet_segmentation.hdf5')
    i=1
    for imgname in imgs:
        image_rgb = (np.array(cv2.imread(imgname, 0))).astype(np.float32)
        image = np.expand_dims(image_rgb, axis=-1) / 255
        net_in = np.zeros((1, 240, 320, 1), dtype=np.float32)
        net_in[0] = image

       # midname = imgname[imgname.rindex("/") + 1:imgname.rindex(".") + 1]

        imgs_mask_test = model.predict(net_in)[0]

        img = imgs_mask_test

        img = array_to_img(img)
        img.save(model_path+"/"+"predicted"+"/"+str(i)+".tiff")
        i=i+1;

    K.clear_session()