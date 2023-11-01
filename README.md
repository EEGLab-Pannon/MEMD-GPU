# MEMD-GPU
This repository contains our CUDA implementation of the Multivariate Empirical Mode Decomposition (MEMD) algorithm. The original algorithm is described in  

- Rehman, N.; Mandic, D.P. Multivariate empirical mode decomposition. Proc. R. Soc. A Math. Phys. Eng. Sci. 2010, 466, 1291–1302, doi:10.1098/rspa.2009.0502.

and implemented in MATLAB available at [https://www.commsp.ee.ic.ac.uk/~mandic/research/memd/memd_version_2.zip](https://www.commsp.ee.ic.ac.uk/~mandic/research/memd/memd_version_2.zip)

Our code achieved up to 430x speedup over an 8-core CPU MATLAB implementatation for 32-channel EEG datasets and 180x speedup for 128-channel EEG datasets on an NVIDIA V100 GPU, reducing the execution time from hours to seconds and days to minutes.

## Compiling and usage instructions  

The supplied code has been tested in Linux and Windows environments. On Linux, compile the code as follows after changing the -arch parameter to match your target CUDA device architecture. This example assumes Volta architecture, i.e. compute capability 7.0. 

`nvcc -arch=sm_70 -Xcompiler -fopenmp -lcublas -lcusparse -lcurand ./sample_synthetic_signal.cu ./memReducedEpochs_dev.cu -o CUDA_MEMD `

This is a single file implementation. The main() function illustrated how to use and execute the cuda MEMD implementation. More detailed examples will be provided in the near future. 

We have also included a MATLAB version that one can integrate into MATLAB scripts. Consult the assistMatlabCuda.m file for MATLAB compilation and usage details. 

More extensive documentation will be added here soon.
