# Nystrom Clustering
Final Project for MATH123 in Tufts MSCS Fall 2021


### Notes: 11/23/21
* All methods are currently functional.
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
