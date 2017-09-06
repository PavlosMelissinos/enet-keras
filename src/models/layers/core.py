# coding=utf-8
def interp(x, shape):
    from keras.backend import tf as ktf
    new_height, new_width = shape
    resized = ktf.image.resize_images(x, [new_height, new_width],
                                      align_corners=True)
    return resized