import cv2
import matplotlib.pyplot as plt 
import numpy as np

def split(img):
    fig = plt.figure(figsize=(10, 10), dpi=72)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img[:,:,0])
    ax1.set_title("Blue")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(img[:,:,1])
    ax2.set_title("Green")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(img[:,:,2])
    ax3.set_title("Red")
    
    plt.title("Channel Split Images")
    plt.show()

def composite(img):
    x,y=img.shape[0],img.shape[1]
    blue=img[:,:,0]
    green=img[:,:,1]
    red=img[:,:,2]

    blue=(blue+np.random.randint(100,size=(x,y)))/255
    green=(green+np.random.randint(100,size=(x,y)))/255
    red=(red+np.random.randint(100,size=(x,y)))/255

    new_image=np.stack([blue,green,red],axis=2)
    
    plt.imshow(new_image)
    plt.title("Composite Image")
    plt.show()

if __name__=='__main__':
    img=cv2.imread("Lab_1/images/images.jpeg")

    print("Splitting into Channels")
    split(img)

    print("Image Composition")
    composite(img)