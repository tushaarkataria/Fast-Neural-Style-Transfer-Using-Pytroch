import numpy as np
from skimage import io
from skimage.transform import resize

def getStyleImage(index):
    if(index==0):
        styleImage   = io.imread('styles/starry_night.jpg')
        styleImage   = styleImage/np.max(styleImage) 
    elif(index==1):
        styleImage   = io.imread('styles/the_scream.jpg')
        styleImage   = styleImage/np.max(styleImage) 
    elif(index==2):
        styleImage   = io.imread('styles/udnie.jpg')
        styleImage   = styleImage/np.max(styleImage) 
    elif(index==3):
        styleImage   = io.imread('styles/wave.jpg')
        styleImage   = styleImage/np.max(styleImage) 
    elif(index==4):
        styleImage   = io.imread('styles/mosiac.jpg')
        styleImage   = styleImage/np.max(styleImage) 
    elif(index==5):
        styleImage   = io.imread('styles/la_muse.jpg')
        styleImage   = styleImage/np.max(styleImage) 
    elif(index==6):
        styleImage   = io.imread('styles/candy.jpg')
        styleImage   = styleImage/np.max(styleImage) 
    return styleImage

