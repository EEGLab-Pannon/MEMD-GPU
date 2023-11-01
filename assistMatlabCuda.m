mexcuda -v '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\lib\x64' -lcublas -lcusparse memdMatlabCuda.cu
%%
numDirVector = int32(64); % input2
numIMFs = int32(9); % input3
maxiter = int32(10); % input4

data = single(data');
[dim1, dim2] = size(data);
signalLength = int32(max(dim1, dim2));
signalDim = int32(min(dim1, dim2)); % input1
samplesIndex = gpuArray(int32((0:(signalLength-1)))); % input6
allPrimesSet = int32(primes(2000));
primesSet = gpuArray(allPrimesSet(1:signalDim)); % input7
outputData = gpuArray(single(zeros(signalLength, signalDim, numIMFs))); % input8
data = gpuArray(data); % input5

IMFs = memdMatlabCuda(signalDim, numDirVector, numIMFs, maxiter, data, samplesIndex, primesSet, outputData);
modes = gather(IMFs);