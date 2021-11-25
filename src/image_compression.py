import cv2
import numpy as np
import matplotlib.pyplot as mplt
from PIL import Image, ImageFilter
from itertools import chain
from numpy.lib.index_tricks import nd_grid
from scipy.sparse import *
from scipy.sparse.csr import csr_matrix
from skimage import filters

sparsity = lambda img: round(np.count_nonzero(img==0)/img.size,4)*100

def avg_diff_transform(img: np.ndarray, axis: str)-> np.ndarray:
    if axis == 2: img = np.swapaxes(img,1,2)
    n = img.shape[2]
    if n <= 1: return img
    even_cols, odd_cols = img[:,:,np.arange(0,n-1,2)], img[:,:,np.arange(1,n,2)]
    avgs = (even_cols + odd_cols)/2; diffs = (even_cols - avgs)
    transformed_img = np.concatenate([avgs, diffs], 2)
    ncols = avgs.shape[2] if avgs.shape[2] % 2 == 0 else avgs.shape[2] - 1
    transformed_img[:,:,:ncols] = avg_diff_transform(transformed_img[:,:,:ncols], 1)
    return transformed_img if axis == 1 else np.swapaxes(transformed_img,1,2)

def compress_image(img: np.ndarray, e: float=0.0)-> np.ndarray:
    img = np.asarray(img)
    sparse_img = avg_diff_transform(avg_diff_transform(img, 1),2)
    sparse_img[sparse_img <= e] = 0
    return sparse_img, sparsity(sparse_img)

def inv_avg_diff_transform(img: np.ndarray, axis: int, chunk_size: int=1)-> np.ndarray:
    if chunk_size == 2 and axis == 1: img = np.swapaxes(img,1,2)
    elif chunk_size >= img.shape[2]: return img if axis == 1 else np.swapaxes(img,1,2)

    avgs, diffs = img[:,:,:chunk_size], img[:,:,chunk_size:2*chunk_size]

    evens = avgs + diffs
    odds = 2*avgs - evens
    left = np.concatenate([evens, odds], 2)
    alt_idxs = list(chain(*[(i, i+chunk_size) for i in range(chunk_size)]))

    transformed_img = np.concatenate([left[:,:,alt_idxs], img[:,:,2*chunk_size:]],2)
    return inv_avg_diff_transform(transformed_img, 2, chunk_size*2)

def decompress_image(compressed_img: np.ndarray)-> np.ndarray:
    compressed_img = np.asarray(compressed_img)
    img = inv_avg_diff_transform(inv_avg_diff_transform(compressed_img, 2), 1)
    return img, sparsity(img)

def compressed_kmeans(img: np.ndarray)-> None:
    timg = np.float32(img).reshape(-1,3)
    k, attempts, criteria = 3, 10, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(timg,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    print(centers)
    result = centers[labels.flatten()]
    result = result.reshape((img.shape))
    return result, centers

def to_Image(A: np.ndarray)-> Image:
    img = None
    try: img = Image.fromarray((A * 1).astype(np.uint8)).convert('RGB')
    except KeyError: Image.fromarray((A.T * 1).astype(np.uint8)).convert('RGB')
    finally: return img

def closest_power_of_two(target: int)-> int:
    count = -1
    print(target)
    while target > 0:
        target >>= 1; count += 1
    return 2**count

def preprocess(A: np.ndarray)-> np.ndarray:
   return (A - np.mean(A, axis=0))

if __name__ == "__main__":
    
    format = 'HSV'
    #image = Image.open('../data/Ridiculousness.jpg').convert(format)
    # image = Image.open('../data/freeman.png').convert(format)
    image = Image.open('../data/circ.png').convert(format)
    #image = Image.open('../data/dots.png').convert(format)
    #image = Image.open('../data/hands.png').convert(format)

    #image = image.filter(ImageFilter.GaussianBlur(radius = 10))
    image_arr = np.asarray(image, dtype=float).T

    if format == 'L': image_arr = np.stack([image_arr, image_arr, image_arr], 0)
    dim = closest_power_of_two(min(image_arr.shape[1:]))
    image_arr = image_arr[:,:dim,:dim]
    print(f"Original {image_arr.shape} image ({sparsity(image_arr)}% sparse)")
    image = Image.fromarray((preprocess(image_arr).T * 1).astype(np.uint8)).convert(format)
    
    sparse_image_arr, sparseness = compress_image(image_arr)
    #sparse_image_arr = filters.scharr(sparse_image_arr)
    print(f"Compressed {sparse_image_arr.shape} image ({sparseness}% sparse)")
    sparse_image = Image.fromarray((preprocess(sparse_image_arr).T * 1).astype(np.uint8)).convert(format)

    dense_image_arr, sparseness = decompress_image(sparse_image_arr)
    # #dense_image_arr = filters.scharr(dense_image_arr)
    # print(f"Decompressed {dense_image_arr.shape} image ({sparseness}% sparse)")
    # dense_image = Image.fromarray((preprocess(dense_image_arr).T * 1).astype(np.uint8)).convert(format)
        
    # image.show(title='ORIGINAL')
    # if input("Continue? ") != 'y': exit()
    # sparse_image.show(title='SPARSE')
    # if input("Continue? ") != 'y': exit()
    #dense_image.show(title='DENSE')

    kimage, centers = compressed_kmeans(dense_image_arr)#sparse_image_arr[:, -dim:, -dim:])
    # #kimage = decompress_image(compressed_kimage)
    #kimage = Image.fromarray((kimage.T * 1).astype(np.uint8)).convert('RGB')

    cv2.imshow('k', kimage.T)
    # mplt.scatter(centers[:,0], centers[:,1], c='r')
    # mplt.scatter(centers[:,1], centers[:,2], c='b')
    # mplt.scatter(centers[:,0], centers[:,2], c='g')
    # mplt.show()
    # exit()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

   # image.show(); sparse_image.show(); kimage.show()
    
