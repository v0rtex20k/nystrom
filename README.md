# Nystrom Clustering
Final Project for MATH123 in Tufts MSCS Fall 2021

### Notes 11/24/21
* Now, all methods are mostly functional, with some loss upon decompress
* It seems most of the relevant edges/information can be found in
  `sparse_image_arr[:,-dim, -dim]`, which is very sparse.
* cv2.kmeans **sucks** - I'm sticking with what I have

* I think my plan of attack is
    1. Set up live video feed, 1024 x 1024 box where users put up their
       hand and start making gestures
    2. Capture every n-th frame apply a skin filter, compress it (?), then  
       pass it to nystrom kmeans
    3. Train on this dataset: http://www.massey.ac.nz/~albarcza/gesture_dataset2012.html from ASL paper
    4. Try and guess the user's gesture based on similarity to
       other gesture images


### Notes: 11/23/21
* ~~All methods are currently functional.~~
* Ng seems to be almost perfect! Pretty fast too.
* Fast is indeed faster, but seems extremely dependent on l and r params.
  Seems to sacrifice accuracy for speed. Experimentally, l = r ~ m/4 works.
* Standard is neither the most accurate nor the fastest - weak sauce.
* Upon testing with large Cali dataset (20000 data-points), Fast proves
  to be significantly faster. Neither Ng nor fast is as accurate. ~~I think
  this is the same problem we saw in the HW, and is inherent to KMeans
  rather than Nystrom - overlapping clusters cannot be reliably detected.
  Perhaps worth looking into alternative clustering algorithm?~~
  This is incorrect, since raw KMeans works. Seems Nystrom is to blame.
