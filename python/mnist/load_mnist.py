#!/home/mql/anaconda3/bin/python3

import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist(path,kind='train'):
    labels_path=os.path.join(path,
                             '%s-labels-idx1-ubyte'
                             % kind)

    images_path=os.path.join(path,
                             '%s-images-idx3-ubyte'
                             % kind)

    with open(labels_path,'rb') as lbpath:
        magic,n=struct.unpack('>II',lbpath.read(8))
        labels=np.fromfile(lbpath,dtype=np.uint8)

    with open(images_path,'rb') as imgpath:
        magic,num,rows,cols=struct.unpack('>IIII',imgpath.read(16))
        images=np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels),784)
    return images,labels

if __name__ == '__main__':
	images,labels=load_mnist('./','train')
	#plt.figure()
	img=images[0].reshape(28,28)
	plt.imshow(img,cmap='Greys',interpolation='nearest')
	#plt.plot(img)
	plt.show()
