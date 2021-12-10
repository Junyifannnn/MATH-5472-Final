# MATH-5472-Final

MATLAB implementation of the paper : 'Weighted Low Rank Matrix Approximation and Acceleration'.

## Code

This repository contains codes for following methods

- PGD based methods:
    - baseline.m
    
    - Nesterov.m
    
    - Anderson.m

    - Anderson_regularization.m

- ALS based methods:
    
    - baseline_ALS.m (Only appears in paper instead of the report)
    
    - baseline_ALS_sparse.m

    - ALS_Nesterov.m

    - ALS_Anderson.m

## Data

For PGD based methods, we use simulated data. And for ALS based methods, we use real world dataset MovieLens dataset from [this link.](https://grouplens.org/datasets/movielens/)

## Usage of the code

```
main.m
```

In this file I give each part corresponding comments of reproducing the corresponding result in the paper. You just need to uncomment the corresponding part then run this file.

