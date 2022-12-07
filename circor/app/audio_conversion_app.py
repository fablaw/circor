import numpy as np
import base64


def image_to_dict(audio_array, dtype='uint8', encoding='utf-8'):
    '''
    Convert an ndarray representing a batch of images into a compressed string
    ----------
    Parameters
    ----------
    imgArray: a np array representing an image
    ----------
    Returns
    ----------
    dict(image: str,
         height: int,
         width: int,
         channel: int)
    '''
    # Get current shape, only for single image
    if audio_array.ndim < 2 or audio_array.ndim > 4:
        raise TypeError
    elif audio_array.ndim < 3:
        audio_array.reshape(*audio_array.shape, 1)
    elif audio_array.ndim < 4:
        size = 1
        height, width, channel = audio_array.shape
    elif audio_array.ndim >= 4:
        size, height, width, channel = audio_array.shape
    # Ensure uint8
    audio_array = audio_array.astype(dtype)
    # Flatten image
    audio_array = audio_array.reshape(size * height * width * channel)
    # Encode in b64 for compression
    audio_array = base64.b64encode(audio_array)
    # Prepare image for POST request, ' cannot be serialized in json
    audio_array = audio_array.decode(encoding).replace("'", '"')
    api_dict = {'audio': audio_array, 'size': size, 'height': height,
                'width': width, 'channel': channel}
    return api_dict
