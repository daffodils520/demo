MS2NMF – NMF Demo (Submission Version)

This folder provides a minimal demo for reproducing the
Non-negative Matrix Factorization (NMF) step used in the MS2NMF framework.

The demo is intended for reproducibility only.

----------------------------------------
Files
----------------------------------------

1. demo_optimized_fragment_matrix.csv

An example optimized fragment–intensity matrix used as the direct input for NMF.

- Rows correspond to samples (or MS/MS features)
- Columns correspond to fragment m/z features
- Values represent optimized fragment intensities

This matrix represents the final output of the preprocessing workflow
described in the manuscript and Supporting Information.

2. run_nmf_demo.py

A minimal Python script that:
- Loads the optimized fragment matrix
- Performs NMF with a fixed number of components
- Outputs the W and H matrices
- Generates simple visualizations of the NMF results

----------------------------------------
Requirements
----------------------------------------

Python >= 3.8

Required packages:
- numpy
- pandas
- scikit-learn
- matplotlib
- scipy

----------------------------------------
How to run
----------------------------------------

From the repository root directory, run:

python demo/run_nmf_demo.py

The results will be saved to:

demo/NMF_results/

----------------------------------------
Notes
----------------------------------------

- This demo reproduces only the NMF factorization step.
- It does not include MS/MS preprocessing, matrix construction, or optimization.
- Full methodological details are described in the manuscript and Supporting Information.
