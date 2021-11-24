import cv2
import numpy as np
from PIL import Image
from itertools import chain

sparsity = lambda img: round(np.count_nonzero(img == 0)/img.size,4)*100

def avg_diff_transform(img: np.ndarray, axis: int)-> np.ndarray:
    if axis == 1: img = np.abs(np.swapaxes(img,0,1))
    n = img.shape[1]
    if n == 1: return img
    even_cols, odd_cols = img[:,np.arange(0,n-1,2),:], img[:,np.arange(1,n,2),:]
    avgs = avg_diff_transform((even_cols + odd_cols)/2, 0) # only swap on final bounce-back
    if even_cols.shape[1] == avgs.shape[1] + 1:
        avgs = np.append(avgs, np.mean(avgs, axis=1)[:,np.newaxis,:], axis=1) # edge case
    transformed_img = np.hstack((avgs, (even_cols - avgs)))
    return transformed_img if axis == 0 else np.swapaxes(transformed_img,0,1)

def compress_image(img: np.ndarray, e: float=50.0)-> np.ndarray:
    img = np.asarray(img)
    sparse_img = avg_diff_transform(avg_diff_transform(img, 0),1)
    sparse_img[sparse_img <= e] = 0
    return sparse_img, sparsity(sparse_img)

def inv_avg_diff_transform(img: np.ndarray, axis: int, chunk_size: int=2)-> np.ndarray:
    if axis == 1: img = np.abs(np.swapaxes(img,0,1))
    n = img.shape[1]
    if chunk_size > n//2: return img if axis == 0 else np.swapaxes(img,0,1)
    elif chunk_size == 4: chunk_size = 5
    print("STARTING w/ IMG:\t", img.shape, ", chunk_size = ", chunk_size)
    avgs, diffs = img[:,:chunk_size,:], img[:,chunk_size:2*chunk_size,:]
    print(chunk_size, 2*chunk_size, img.shape)
    sums = avgs + diffs
    rems = 2*avgs - sums
    print("SUMS:\t", sums.shape)
    print("REMS:\t", rems.shape)
    outer, idxs = np.hstack((sums, rems)), np.arange(chunk_size)
    print("OUTER:\t", outer.shape)
    img = np.hstack((outer, img[:,2*chunk_size:,:]))
    img = img if axis == 0 else np.swapaxes(img,0,1)
    print("\tRECURSING w/ IMG:\t", img.shape)
    return inv_avg_diff_transform(img, 0, chunk_size*2)
    
    # if axis == 1: img = np.abs(np.swapaxes(img,0,1))
    # n = img.shape[1]
    # if n == 1: return img if axis == 0 else np.swapaxes(img,0,1)
    # even_cols, odd_cols = img[:,np.arange(0,n-1,2),:], img[:,np.arange(1,n,2),:]
    # avgs = avg_diff_transform((even_cols + odd_cols)/2, axis)
    # if avgs.shape[1] % 2 == 0: return avgs
    # transformed_img = np.hstack((avgs, (even_cols - avgs)))
    # return transformed_img if axis == 0 else np.swapaxes(transformed_img,0,1)

def decompress_image(compressed_img: np.ndarray)-> np.ndarray:
    compressed_img = np.asarray(compressed_img)
    img = inv_avg_diff_transform(inv_avg_diff_transform(compressed_img, 1), 0)
    return img, sparsity(img)

def compressed_kmeans(img: np.ndarray)-> None:
    img = np.float32(img)
    k, attempts, criteria = 10, 10, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(img,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)

    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

'''
TODO:
    - This seems flawed. Perhaps a better approach is img_to_graph?
      See example here: https://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html#sphx-glr-auto-examples-cluster-plot-segmentation-toy-py
      Just substitute spectral clustering for nystrom and only keep labels...

    - I think compression works? But decompressing is a nightmare.
'''


if __name__ == "__main__":

    image = Image.open('../data/Ridiculousness.jpg').convert('HSV')
    image_arr = np.asarray(image)
    print('--->', image_arr.shape)
    print(f"Original {image_arr.shape} image ({sparsity(image_arr)}% sparse)")
    
    sparse_image_arr, sparseness = compress_image(image_arr)
    print(f"Compressed {sparse_image_arr.shape} image ({sparseness}% sparse)")
    sparse_image = Image.fromarray((sparse_image_arr * 1).astype(np.uint8)).convert('RGB')

    dense_image_arr, sparseness = decompress_image(sparse_image_arr)
    print(f"Decompressed {dense_image_arr.shape} image ({sparseness}% sparse)")
    dense_image = Image.fromarray((dense_image_arr * 1).astype(np.uint8)).convert('RGB')
        
    image.show(); sparse_image.show(); dense_image.show()
    exit()

    kimage = compressed_kmeans(image_arr)
    #kimage = decompress_image(compressed_kimage)
    kimage = Image.fromarray((kimage * 1).astype(np.uint8)).convert('RGB')
    
    image.show(); sparse_image.show(); kimage.convert('RGB').show()
    
