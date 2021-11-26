import numpy as np
from skimage import io
from skimage.transform import resize

def getStyleImage(index):
    if(index==0):
        styleImage   = io.imread('styles/starry_night.jpg')
        #styleImage   = styleImage/np.max(styleImage) 
    elif(index==1):
        styleImage   = io.imread('styles/the_scream.jpg')
        #styleImage   = styleImage/np.max(styleImage) 
    elif(index==2):
        styleImage   = io.imread('styles/udnie.jpg')
        #styleImage   = styleImage/np.max(styleImage) 
    elif(index==3):
        styleImage   = io.imread('styles/wave.jpg')
        #styleImage   = styleImage/np.max(styleImage) 
    elif(index==4):
        styleImage   = io.imread('styles/mosaic.jpg')
        #styleImage   = styleImage/np.max(styleImage) 
    elif(index==5):
        styleImage   = io.imread('styles/la_muse.jpg')
        #styleImage   = styleImage/np.max(styleImage) 
    elif(index==6):
        styleImage   = io.imread('styles/candy.jpg')
        #styleImage   = styleImage/np.max(styleImage) 
    elif(index==7):
        styleImage   = io.imread('styles/composition_vii.jpg')
        #styleImage   = styleImage/np.max(styleImage) 
    elif(index==8):
        styleImage   = io.imread('styles/SampleStyle-2.jpg')
        #styleImage   = styleImage/np.max(styleImage) 
    elif(index==9):
        styleImage   = io.imread('styles/SampleStyle-1.jpg')
        #styleImage   = styleImage/np.max(styleImage) 
    elif(index==10):
        styleImage   = io.imread('styles/SampleStyle-4.jpg')
        #styleImage   = styleImage/np.max(styleImage) 
    return styleImage

