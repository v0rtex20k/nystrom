import cv2
import numpy as np
from PIL import Image
from itertools import chain

sparsity = lambda img: round(np.count_nonzero(img == 0)/img.size,4)*100

def avg_diff_transform(img: np.ndarray, axis: str)-> np.ndarray:
    if axis == 2: img = np.swapaxes(img,1,2)
    n = img.shape[2]
    if n <= 1: return img
    even_cols, odd_cols = img[:,:,np.arange(0,n-1,2)], img[:,:,np.arange(1,n,2)]
    avgs = (even_cols + odd_cols)/2; diffs = (even_cols - avgs)
    transformed_img = np.concatenate([avgs, diffs], 2)
    ncols = avgs.shape[2] if avgs.shape[2] % 2 == 0 else avgs.shape[2] - 1
    transformed_img[:,:,:ncols] = avg_diff_transform(transformed_img[:,:,:ncols], 1) # only swap on final bounce-back
    return transformed_img if axis == 1 else np.swapaxes(transformed_img,1,2)

def compress_image(img: np.ndarray, e: float=50.0)-> np.ndarray:
    img = np.asarray(img)
    sparse_img = avg_diff_transform(avg_diff_transform(img, 1),2)
    sparse_img[sparse_img <= e] = 0
    return sparse_img, sparsity(sparse_img)

def inv_avg_diff_transform(img: np.ndarray, axis: int, chunk_size: int=1)-> np.ndarray:
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

def to_Image(A: np.ndarray)-> Image:
    return Image.fromarray((A * 1).astype(np.uint8)).convert('RGB')

if __name__ == "__main__":
    # image = np.array([[64,2,3,61,60,6,7,57],
    #                   [9,55,54,12,13,51,50,16],
    #                   [17,47,46,20,21,43,42,24], 
    #                   [40,26,27,37,36,30,31,33], 
    #                   [32,34,35,29,28,38,39,25],
    #                   [41,23,22,44,45,19,18,48],
    #                   [49,15,14,52,53,11,10,56],
    #                   [8,58,59,5,4,62,63,1]])
    # image = np.stack([image, image, image], axis=0)
    # print(f"Original {image.shape} image ({sparsity(image)}% sparse)")
    # timage = avg_diff_transform(avg_diff_transform(image, 0), 1)
    # print(timage)
    # print(f"Compressed {timage.shape} image ({sparsity(timage)}% sparse)")
    # #to_Image(image).show(); to_Image(timage).show()
    # exit()
    image = Image.open('../data/Ridiculousness.jpg').convert('HSV')
    image_arr = np.asarray(image)
    print('--->', image_arr.shape)
    print(f"Original {image_arr.shape} image ({sparsity(image_arr)}% sparse)")
    
    sparse_image_arr, sparseness = compress_image(image_arr)
    print(f"Compressed {sparse_image_arr.shape} image ({sparseness}% sparse)")
    sparse_image = Image.fromarray((sparse_image_arr * 1).astype(np.uint8)).convert('RGB')
    
    # dense_image_arr, sparseness = decompress_image(sparse_image_arr)
    # print(f"Decompressed {dense_image_arr.shape} image ({sparseness}% sparse)")
    # dense_image = Image.fromarray((dense_image_arr * 1).astype(np.uint8)).convert('RGB')
        
    image.show(); sparse_image.show();# dense_image.show()
    exit()

    kimage = compressed_kmeans(image_arr)
    #kimage = decompress_image(compressed_kimage)
    kimage = Image.fromarray((kimage * 1).astype(np.uint8)).convert('RGB')
    
    image.show(); sparse_image.show(); kimage.convert('RGB').show()
    
