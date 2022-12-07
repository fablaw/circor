
import numpy as np
import base64

def image_from_dict(api_dict, dtype='uint8', encoding='utf-8'):
    '''
    Convert an dict representing a batch of images into a ndarray
    ----------
    Parameters
    ----------
    api_dict: a dict(image, height, width, channel) representing an image
    dtype: target data type for ndarray
    encoding: encoding used for image string
    ----------
    Returns
    ----------
    ndarray of shape (size, height, width, channel)
    '''
    # Decode image string
    img = base64.b64decode(bytes(api_dict.get('image'), encoding))
    # Convert to np.ndarray and ensure dtype
    img = np.frombuffer(img, dtype=dtype)
    # Reshape to original shape
    img = img.reshape((api_dict.get('size'),
                      api_dict.get('height'),
                      api_dict.get('width'),
                      api_dict.get('channel')))
    return img
