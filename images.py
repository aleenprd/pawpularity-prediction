#%%
import cv2

im = cv2.imread('data/train/00a1ae8867e0bb89f061679e1cf29e80.jpg')

print(type(im))
# <class 'numpy.ndarray'>

print(im.shape)
print(type(im.shape))
# (225, 400, 3)
# <class 'tuple'>
# %%
