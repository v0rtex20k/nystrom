# Nystrom Clustering
## Final Project for MATH123 in Tufts MSCS Fall 2021
### Members: Victor Arsenescu, Ryan Ghayour, Pippa Hodgkins

&nbsp;

Simply clone the repo and run `python3 src/mnist.py` (for *MNIST* dataset) or `python3 src/small_datasets.py` (for *California Housing* dataset). These give a cleaner interface for users to run the algorithms, but the math itself is implemented in `src/algos.py` and correspond to the algorithms discussed in the paper as follows

- `algo0` <---> **Vanilla KMeans** (blue line)
- `algo1` <---> **Spectral Clustering w/o Nystrom** (SCwoN, red line)
- `algo3` <---> **Spectral Clustering w/ Nystrom** (SCN, green line)

Thank you for a wonderful semester - Happy Holidays! 
