from PIL import Image
import numpy as np
def preprocess(image):
    img=image.resize((224,224))
    return np.array(img)/255.0
