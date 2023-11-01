
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <tuple>
#include <omp.h>
#include <dirent.h>

#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)
int THREADS_PER_BLOCK = 256;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

//detect extrema (deprecated)
template <typename coord_t, typename real_t>
__global__ void find_extrema_shfl(const coord_t* d_multiChannelIndex, const real_t* d_ProjectSignals, coord_t* d_sparseMaxIndex, real_t* d_sparseMaxValue, coord_t* d_sparseMinIndex, real_t* d_sparseMinValue, coord_t* d_sparseMaxFlag, coord_t* d_sparseMinFlag, size_t SignalLength) {

    int channelIndex = blockIdx.y;
    int channelElementsIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int allElementsIndex = blockIdx.y * SignalLength + blockIdx.x * blockDim.x + threadIdx.x;

    int warpFlag = channelElementsIndex / 32;

    if ((channelElementsIndex - 2 * warpFlag) < SignalLength)
    {
        real_t value = d_ProjectSignals[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)];
        coord_t coord = d_multiChannelIndex[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)];

        real_t up = __shfl_up_sync(0xffffffff, value, 1);
        real_t down = __shfl_down_sync(0xffffffff, value, 1);
        if (value > up && value > down)
        {
            d_sparseMaxIndex[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)] = coord;
            d_sparseMaxValue[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)] = value;
            d_sparseMaxFlag[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)] = 1;
        }


        if (value < up && value < down)
        {
            d_sparseMinIndex[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)] = coord;
            d_sparseMinValue[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)] = value;
            d_sparseMinFlag[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)] = 1;
        }
        // for temporary, set edge points as it were 
        if (channelElementsIndex == 0)
        {
            d_sparseMaxIndex[blockIdx.y * SignalLength] = d_multiChannelIndex[blockIdx.y * SignalLength];
            d_sparseMaxValue[blockIdx.y * SignalLength] = d_ProjectSignals[blockIdx.y * SignalLength];
            d_sparseMaxFlag[blockIdx.y * SignalLength] = 1;

            d_sparseMinIndex[blockIdx.y * SignalLength] = d_multiChannelIndex[blockIdx.y * SignalLength];
            d_sparseMinValue[blockIdx.y * SignalLength] = d_ProjectSignals[blockIdx.y * SignalLength];
            d_sparseMinFlag[blockIdx.y * SignalLength] = 1;
        }

        if ((channelElementsIndex - 2 * warpFlag) == (SignalLength - 1))
        {
            d_sparseMaxIndex[blockIdx.y * SignalLength + (SignalLength - 1)] = d_multiChannelIndex[blockIdx.y * SignalLength + (SignalLength - 1)];
            d_sparseMaxValue[blockIdx.y * SignalLength + (SignalLength - 1)] = d_ProjectSignals[blockIdx.y * SignalLength + (SignalLength - 1)];
            d_sparseMaxFlag[blockIdx.y * SignalLength + (SignalLength - 1)] = 1;

            d_sparseMinIndex[blockIdx.y * SignalLength + (SignalLength - 1)] = d_multiChannelIndex[blockIdx.y * SignalLength + (SignalLength - 1)];
            d_sparseMinValue[blockIdx.y * SignalLength + (SignalLength - 1)] = d_ProjectSignals[blockIdx.y * SignalLength + (SignalLength - 1)];
            d_sparseMinFlag[blockIdx.y * SignalLength + (SignalLength - 1)] = 1;
        }
    }

}

template <typename coord_t, typename real_t>
__global__ void find_extrema_shfl_max(const coord_t* d_multiChannelIndex, const real_t* d_ProjectSignals, coord_t* d_sparseMaxFlag, size_t SignalLength) {

    int channelIndex = blockIdx.y;
    int channelElementsIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int allElementsIndex = blockIdx.y * SignalLength + blockIdx.x * blockDim.x + threadIdx.x;

    int warpFlag = channelElementsIndex / 32;

    if ((channelElementsIndex - 2 * warpFlag) < SignalLength)
    {
        real_t value = d_ProjectSignals[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)];
        coord_t coord = d_multiChannelIndex[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)];

        real_t up = __shfl_up_sync(0xffffffff, value, 1);
        real_t down = __shfl_down_sync(0xffffffff, value, 1);
        if (value > up && value > down)
        {
            d_sparseMaxFlag[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)] = 1;
        }

        // for temporary, set edge points as it were 
        if (channelElementsIndex == 0)
        {
            d_sparseMaxFlag[blockIdx.y * SignalLength] = 1;
            d_sparseMaxFlag[blockIdx.y * SignalLength + SignalLength - 1] = 1;
        }

        //deprecated
        //if ((channelElementsIndex - 2 * warpFlag) == (SignalLength - 1))
        //{
        //    d_sparseMaxFlag[blockIdx.y * SignalLength + (SignalLength - 1)] = 1;
        //}
    }
}

template <typename coord_t, typename real_t>
__global__ void find_extrema_shfl_min(const coord_t* d_multiChannelIndex, const real_t* d_ProjectSignals, coord_t* d_sparseMinFlag, size_t SignalLength) {

    int channelIndex = blockIdx.y;
    int channelElementsIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int allElementsIndex = blockIdx.y * SignalLength + blockIdx.x * blockDim.x + threadIdx.x;

    int warpFlag = channelElementsIndex / 32;

    if ((channelElementsIndex - 2 * warpFlag) < SignalLength)
    {
        real_t value = d_ProjectSignals[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)];
        coord_t coord = d_multiChannelIndex[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)];

        real_t up = __shfl_up_sync(0xffffffff, value, 1);
        real_t down = __shfl_down_sync(0xffffffff, value, 1);

        if (value < up && value < down)
        {
            d_sparseMinFlag[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)] = 1;
        }
        // for temporary, set edge points as it were 
        if (channelElementsIndex == 0)
        {
            d_sparseMinFlag[blockIdx.y * SignalLength] = 1;
            d_sparseMinFlag[blockIdx.y * SignalLength + SignalLength - 1] = 1;
        }
    }
}

__global__ void prescan_arbitrary(int* output, int* input, int n, int powerOfTwo)
{
    extern __shared__ int temp[];// allocated on invocation
    int threadID = threadIdx.x;

    int ai = threadID;
    int bi = threadID + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);


    if (threadID < n) {
        temp[ai + bankOffsetA] = input[ai];
        temp[bi + bankOffsetB] = input[bi];
    }
    else {
        temp[ai + bankOffsetA] = 0;
        temp[bi + bankOffsetB] = 0;
    }


    int offset = 1;
    for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (threadID == 0) {
        temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0; // clear the last element
    }

    for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    if (threadID < n) {
        output[ai] = temp[ai + bankOffsetA];
        output[bi] = temp[bi + bankOffsetB];
    }
}

__global__ void prescan_large(int* output, int* input, int n, int* sums) {
    extern __shared__ int temp[];

    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int blockOffset = blockID * n;

    int ai = threadID;
    int bi = threadID + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = input[blockOffset + ai];
    temp[bi + bankOffsetB] = input[blockOffset + bi];

    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    __syncthreads();


    if (threadID == 0) {
        sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }

    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    output[blockOffset + ai] = temp[ai + bankOffsetA];
    output[blockOffset + bi] = temp[bi + bankOffsetB];
}

__global__ void add(int* output, int length, int* n) {
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int blockOffset = blockID * length;

    output[blockOffset + threadID] += n[blockID];
}

__global__ void add(int* output, int length, int* n1, int* n2) {
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int blockOffset = blockID * length;

    output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}

__global__ void find_extrema_shfl_max_4matlab(const int* d_multiChannelIndex, const float* d_ProjectSignals, int* d_sparseMaxFlag, int SignalLength) {

    int channelIndex = blockIdx.y;
    int channelElementsIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int allElementsIndex = blockIdx.y * SignalLength + blockIdx.x * blockDim.x + threadIdx.x;

    int warpFlag = channelElementsIndex / 32;

    if ((channelElementsIndex - 2 * warpFlag) < SignalLength)
    {
        float value = d_ProjectSignals[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)];
        int coord = d_multiChannelIndex[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)];

        float up = __shfl_up_sync(0xffffffff, value, 1);
        float down = __shfl_down_sync(0xffffffff, value, 1);
        if (value > up && value > down)
        {
            d_sparseMaxFlag[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)] = 1;
        }

        // for temporary, set edge points as it were 
        if (channelElementsIndex == 0)
        {
            d_sparseMaxFlag[blockIdx.y * SignalLength] = 1;
            d_sparseMaxFlag[blockIdx.y * SignalLength + SignalLength - 1] = 1;
        }

        //deprecated
        //if ((channelElementsIndex - 2 * warpFlag) == (SignalLength - 1))
        //{
        //    d_sparseMaxFlag[blockIdx.y * SignalLength + (SignalLength - 1)] = 1;
        //}
    }
}

void scanLargeDeviceArray(int* d_out, int* d_in, int length, bool bcao, int* d_sums, int* d_incr);
void scanSmallDeviceArray(int* d_out, int* d_in, int length, bool bcao);
void scanLargeEvenDeviceArray(int* d_out, int* d_in, int length, bool bcao, int* d_sums, int* d_incr);

int nextPowerOfTwo(int x) {
    int power = 1;
    while (power < x) {
        power *= 2;
    }
    return power;
}

void scanLargeDeviceArray(int* d_out, int* d_in, int length, bool bcao, int* d_sums, int* d_incr) {
    int remainder = length % (ELEMENTS_PER_BLOCK);
    if (remainder == 0) {
        scanLargeEvenDeviceArray(d_out, d_in, length, bcao, d_sums, d_incr);
    }
    else {
        // perform a large scan on a compatible multiple of elements
        int lengthMultiple = length - remainder;
        scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, bcao, d_sums, d_incr);

        // scan the remaining elements and add the (inclusive) last element of the large scan to this
        int* startOfOutputArray = &(d_out[lengthMultiple]);
        scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder, bcao);

        add << <1, remainder >> > (startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
    }
}

void scanSmallDeviceArray(int* d_out, int* d_in, int length, bool bcao) {
    int powerOfTwo = nextPowerOfTwo(length);

    if (bcao) {
        prescan_arbitrary << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> > (d_out, d_in, length, powerOfTwo);
    }
    else {
        //prescan_arbitrary_unoptimized << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> > (d_out, d_in, length, powerOfTwo);
    }
}

void scanLargeEvenDeviceArray(int* d_out, int* d_in, int length, bool bcao, int* d_sums, int* d_incr) {
    const int blocks = length / ELEMENTS_PER_BLOCK;
    const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

    if (bcao) {
        prescan_large << <blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize >> > (d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
    }
    else {
        //prescan_large_unoptimized << <blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize >> > (d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
    }

    const int sumsArrThreadsNeeded = (blocks + 1) / 2;
    if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
        // perform a large scan on the sums arr
        scanLargeDeviceArray(d_incr, d_sums, blocks, bcao, d_sums, d_incr);
    }
    else {
        // only need one block to scan sums arr so can use small scan
        scanSmallDeviceArray(d_incr, d_sums, blocks, bcao);
    }

    add << <blocks, ELEMENTS_PER_BLOCK >> > (d_out, ELEMENTS_PER_BLOCK, d_incr);

}

template <typename coord_t, typename real_t>
__global__ void select_extrema_max(coord_t* d_sparseMaxFlag, real_t* d_current, coord_t* d_multiDirVecChanSignalIndex, coord_t* d_MaxScanResult,
    real_t* d_compactMaxValue, coord_t* d_compactMaxIndex, size_t SignalLength, size_t SignalDim, size_t NumDirVector, coord_t* d_num_extrema_max)
{
    int dirVecIdx = blockIdx.z;
    int signalDimIdx = blockIdx.y;
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pointsIdx < SignalLength)
    {
        real_t currentValue = d_current[signalDimIdx * SignalLength + pointsIdx];
        coord_t currentIndex = d_multiDirVecChanSignalIndex[dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + pointsIdx];
        coord_t currentFlag = d_sparseMaxFlag[dirVecIdx * SignalLength + pointsIdx];
        coord_t storeLocation = d_MaxScanResult[dirVecIdx * SignalLength + pointsIdx];

        if (currentFlag != 0)
        {
            d_compactMaxValue[dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + storeLocation] = currentValue;
            d_compactMaxIndex[dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + storeLocation] = currentIndex;
        }

        if (pointsIdx == SignalLength - 1)
        {
            if (signalDimIdx == 0)
            {
                d_num_extrema_max[dirVecIdx] = storeLocation + 1;
            }
        }
    }
}

template <typename coord_t, typename real_t>
__global__ void select_extrema_min(coord_t* d_sparseMinFlag, real_t* d_current, coord_t* d_multiDirVecChanSignalIndex,
    coord_t* d_MinScanResult, real_t* d_compactMinValue, coord_t* d_compactMinIndex, size_t SignalLength,
    size_t SignalDim, size_t NumDirVector, coord_t* d_num_extrema_min)
{
    int dirVecIdx = blockIdx.z;
    int signalDimIdx = blockIdx.y;
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pointsIdx < SignalLength)
    {
        real_t currentValue = d_current[signalDimIdx * SignalLength + pointsIdx];
        coord_t currentIndex = d_multiDirVecChanSignalIndex[dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + pointsIdx];
        coord_t currentFlag = d_sparseMinFlag[dirVecIdx * SignalLength + pointsIdx];
        coord_t storeLocation = d_MinScanResult[dirVecIdx * SignalLength + pointsIdx];

        if (currentFlag != 0)
        {
            d_compactMinValue[dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + storeLocation] = currentValue;
            d_compactMinIndex[dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + storeLocation] = currentIndex;
        }

        if (pointsIdx == SignalLength - 1)
        {
            if (signalDimIdx == 0)
            {
                d_num_extrema_min[dirVecIdx] = storeLocation + 1;
            }
        }
    }
}

template <typename coord_t, typename real_t>
__global__ void setBoundaryMax(real_t* d_compactMaxValue, coord_t* d_compactMaxIndex, coord_t* d_MaxScanResult, size_t SignalLength, size_t SignalDim, size_t NumDirVector)
{
    int dirVecIdx = blockIdx.z;
    int signalDimIdx = blockIdx.y;
    int pointsIdx = threadIdx.x;

    if ((pointsIdx == 0) && (d_MaxScanResult[dirVecIdx * SignalLength + SignalLength - 1] > 4))
    {
        real_t slope_max, t_max;

        coord_t storeLocation_max = d_MaxScanResult[dirVecIdx * SignalLength + pointsIdx];
        coord_t loc_max = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + storeLocation_max;

        slope_max = (d_compactMaxValue[loc_max + 2] - d_compactMaxValue[loc_max + 1]) / (d_compactMaxIndex[loc_max + 2] - d_compactMaxIndex[loc_max + 1]);
        t_max = d_compactMaxValue[loc_max + 1] - slope_max * (d_compactMaxIndex[loc_max + 1] - d_compactMaxIndex[loc_max]);

        if (t_max > d_compactMaxValue[loc_max])
        {
            d_compactMaxValue[loc_max] = t_max;
        }
    }

    if ((pointsIdx == 1) && (d_MaxScanResult[dirVecIdx * SignalLength + SignalLength - 1] > 4))
    {

        real_t slope_max, t_max;

        coord_t storeLocation_max = d_MaxScanResult[dirVecIdx * SignalLength + SignalLength - 1];
        coord_t loc_max = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + storeLocation_max;

        slope_max = (d_compactMaxValue[loc_max - 1] - d_compactMaxValue[loc_max - 2]) / (d_compactMaxIndex[loc_max - 1] - d_compactMaxIndex[loc_max - 2]);
        t_max = d_compactMaxValue[loc_max - 1] + slope_max * (d_compactMaxIndex[loc_max] - d_compactMaxIndex[loc_max - 1]);

        if (t_max > d_compactMaxValue[loc_max])
        {
            d_compactMaxValue[loc_max] = t_max;
        }
    }
}

template <typename coord_t, typename real_t>
__global__ void setBoundary(real_t* d_compactMaxValue, real_t* d_compactMinValue, coord_t* d_compactMaxIndex, coord_t* d_compactMinIndex, coord_t* d_MaxScanResult, coord_t* d_MinScanResult, size_t SignalLength, size_t SignalDim, size_t NumDirVector)
{
    int dirVecIdx = blockIdx.z;
    int signalDimIdx = blockIdx.y;
    int pointsIdx = threadIdx.x;

    if ((pointsIdx == 0) && (d_MaxScanResult[dirVecIdx * SignalLength + SignalLength - 1] > 4) && (d_MinScanResult[dirVecIdx * SignalLength + SignalLength - 1] > 4))
    {
        real_t slope_max, t_max, slope_min, t_min;

        coord_t storeLocation_max = d_MaxScanResult[dirVecIdx * SignalLength + pointsIdx];
        coord_t loc_max = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + storeLocation_max;
        coord_t storeLocation_min = d_MinScanResult[dirVecIdx * SignalLength + pointsIdx];
        coord_t loc_min = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + storeLocation_min;

        slope_max = (d_compactMaxValue[loc_max + 2] - d_compactMaxValue[loc_max + 1]) / (d_compactMaxIndex[loc_max + 2] - d_compactMaxIndex[loc_max + 1]);
        t_max = d_compactMaxValue[loc_max + 1] - slope_max * (d_compactMaxIndex[loc_max + 1] - d_compactMaxIndex[loc_max]);
        slope_min = (d_compactMinValue[loc_min + 2] - d_compactMinValue[loc_min + 1]) / (d_compactMinIndex[loc_min + 2] - d_compactMinIndex[loc_min + 1]);
        t_min = d_compactMinValue[loc_min + 1] - slope_min * (d_compactMinIndex[loc_min + 1] - d_compactMinIndex[loc_min]);

        if (t_max > d_compactMaxValue[loc_max])
        {
            d_compactMaxValue[loc_max] = t_max;
        }
        if (t_min < d_compactMinValue[loc_min])
        {
            d_compactMinValue[loc_min] = t_min;
        }
    }

    if ((pointsIdx == 1) && (d_MaxScanResult[dirVecIdx * SignalLength + SignalLength - 1] > 4) && (d_MinScanResult[dirVecIdx * SignalLength + SignalLength - 1] > 4))
    {

        real_t slope_max, t_max, slope_min, t_min;

        coord_t storeLocation_max = d_MaxScanResult[dirVecIdx * SignalLength + SignalLength - 1];
        coord_t loc_max = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + storeLocation_max;
        coord_t storeLocation_min = d_MinScanResult[dirVecIdx * SignalLength + SignalLength - 1];
        coord_t loc_min = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + storeLocation_min;

        slope_max = (d_compactMaxValue[loc_max - 1] - d_compactMaxValue[loc_max - 2]) / (d_compactMaxIndex[loc_max - 1] - d_compactMaxIndex[loc_max - 2]);
        t_max = d_compactMaxValue[loc_max - 1] + slope_max * (d_compactMaxIndex[loc_max] - d_compactMaxIndex[loc_max - 1]);
        slope_min = (d_compactMinValue[loc_min - 1] - d_compactMinValue[loc_min - 2]) / (d_compactMinIndex[loc_min - 1] - d_compactMinIndex[loc_min - 2]);
        t_min = d_compactMinValue[loc_min - 1] + slope_min * (d_compactMinIndex[loc_min] - d_compactMinIndex[loc_min - 1]);

        if (t_max > d_compactMaxValue[loc_max])
        {
            d_compactMaxValue[loc_max] = t_max;
        }
        if (t_min < d_compactMinValue[loc_min])
        {
            d_compactMinValue[loc_min] = t_min;
        }
    }
}

template <typename coord_t, typename real_t>
__global__ void setBoundaryMin(real_t* d_compactMinValue, coord_t* d_compactMinIndex, coord_t* d_MinScanResult, size_t SignalLength, size_t SignalDim, size_t NumDirVector)
{
    int dirVecIdx = blockIdx.z;
    int signalDimIdx = blockIdx.y;
    int pointsIdx = threadIdx.x;

    if ((pointsIdx == 0) && (d_MinScanResult[dirVecIdx * SignalLength + SignalLength - 1] > 4))
    {
        real_t slope_min, t_min;

        coord_t storeLocation_min = d_MinScanResult[dirVecIdx * SignalLength + pointsIdx];
        coord_t loc_min = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + storeLocation_min;

        slope_min = (d_compactMinValue[loc_min + 2] - d_compactMinValue[loc_min + 1]) / (d_compactMinIndex[loc_min + 2] - d_compactMinIndex[loc_min + 1]);
        t_min = d_compactMinValue[loc_min + 1] - slope_min * (d_compactMinIndex[loc_min + 1] - d_compactMinIndex[loc_min]);

        if (t_min < d_compactMinValue[loc_min])
        {
            d_compactMinValue[loc_min] = t_min;
        }
    }

    if ((pointsIdx == 1) && (d_MinScanResult[dirVecIdx * SignalLength + SignalLength - 1] > 4))
    {

        real_t slope_min, t_min;

        coord_t storeLocation_min = d_MinScanResult[dirVecIdx * SignalLength + SignalLength - 1];
        coord_t loc_min = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + storeLocation_min;

        slope_min = (d_compactMinValue[loc_min - 1] - d_compactMinValue[loc_min - 2]) / (d_compactMinIndex[loc_min - 1] - d_compactMinIndex[loc_min - 2]);
        t_min = d_compactMinValue[loc_min - 1] + slope_min * (d_compactMinIndex[loc_min] - d_compactMinIndex[loc_min - 1]);

        if (t_min < d_compactMinValue[loc_min])
        {
            d_compactMinValue[loc_min] = t_min;
        }
    }
}

//set boundary (deprecated)
template <typename coord_t, typename real_t>
__global__ void select_extrema_max_boundary(coord_t* d_sparseMaxFlag, real_t* d_current, coord_t* d_multiDirVecChanSignalIndex, coord_t* d_MaxScanResult,
    real_t* d_compactMaxValue, coord_t* d_compactMaxIndex, size_t SignalLength, size_t SignalDim, size_t NumDirVector, coord_t* d_num_extrema_max)
{
    int dirVecIdx = blockIdx.z;
    int signalDimIdx = blockIdx.y;
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if ((pointsIdx == 0) && (d_MaxScanResult[dirVecIdx * SignalLength + SignalLength - 1] > 4))
    {
        real_t slope_max, t_max;
        coord_t storeLocation = d_MaxScanResult[dirVecIdx * SignalLength + pointsIdx];
        coord_t loc = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + storeLocation;

        real_t num = (d_compactMaxValue[loc + 2] - d_compactMaxValue[loc + 1]);
        coord_t denom = (d_compactMaxIndex[loc + 2] - d_compactMaxIndex[loc + 1]);
        slope_max = num / denom;
        t_max = d_compactMaxValue[loc + 1] - slope_max * (d_compactMaxIndex[loc + 1] - d_compactMaxIndex[loc]);
        if (isinf(t_max))
            printf("inf error-3 in select extrema max \n");

        if (t_max > d_compactMaxValue[loc])
        {
            //if (isinf(t_max))
            //	printf("inf error-2 in select extrema\n");
            d_compactMaxValue[loc] = t_max;
        }
    }

    __threadfence();
    __syncthreads();

    if ((pointsIdx == SignalLength - 1) && (d_MaxScanResult[dirVecIdx * SignalLength + SignalLength - 1] > 4))
    {

        real_t slope_max, t_max;
        coord_t storeLocation = d_MaxScanResult[dirVecIdx * SignalLength + pointsIdx];
        coord_t loc = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + storeLocation;

        //real_t up = d_compactMaxValue[loc - 1] - d_compactMaxValue[loc - 2];
        //real_t down = (real_t)(d_compactMaxIndex[loc - 1] - d_compactMaxIndex[loc - 2]);
        //slope_max = up / (real_t)down;
        slope_max = (d_compactMaxValue[loc - 1] - d_compactMaxValue[loc - 2]) / (d_compactMaxIndex[loc - 1] - d_compactMaxIndex[loc - 2]);
        t_max = d_compactMaxValue[loc - 1] + slope_max * (d_compactMaxIndex[loc] - d_compactMaxIndex[loc - 1]);
        if (isinf(t_max))
            printf("inf error-4 in select extrema max \n");

        if (t_max > d_compactMaxValue[loc])
        {
            d_compactMaxValue[loc] = t_max;
        }
        //if (isinf(d_compactMaxValue[loc]))
        //	printf("inf error-4 in select extrema\n");

    }

}

template <typename coord_t, typename real_t>
__global__ void select_extrema_min_boundary(coord_t* d_sparseMinFlag, real_t* d_current, coord_t* d_multiDirVecChanSignalIndex,
    coord_t* d_MinScanResult, real_t* d_compactMinValue, coord_t* d_compactMinIndex, size_t SignalLength,
    size_t SignalDim, size_t NumDirVector, coord_t* d_num_extrema_min)
{
    int dirVecIdx = blockIdx.z;
    int signalDimIdx = blockIdx.y;
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if ((pointsIdx == 0) && (d_MinScanResult[dirVecIdx * SignalLength + SignalLength - 1] > 4))
    {
        real_t slope_min, t_min;
        coord_t storeLocation = d_MinScanResult[dirVecIdx * SignalLength + pointsIdx];
        coord_t loc = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + storeLocation;

        slope_min = (d_compactMinValue[loc + 2] - d_compactMinValue[loc + 1]) / (d_compactMinIndex[loc + 2] - d_compactMinIndex[loc + 1]);
        t_min = d_compactMinValue[loc + 1] - slope_min * (d_compactMinIndex[loc + 1] - d_compactMinIndex[loc]);
        if (isinf(t_min))
            printf("inf error-3 in select extrema min \n");

        if (t_min < d_compactMinValue[loc])
        {
            //if (isinf(t_min))
            //	printf("inf error-2 in select extrema \n");
            d_compactMinValue[loc] = t_min;
        }
    }

    __threadfence();
    __syncthreads();

    if ((pointsIdx == SignalLength - 1) && (d_MinScanResult[dirVecIdx * SignalLength + SignalLength - 1] > 4))
    {
        real_t slope_min, t_min;
        coord_t storeLocation = d_MinScanResult[dirVecIdx * SignalLength + pointsIdx];
        coord_t loc = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + storeLocation;

        //real_t up = d_compactMinValue[loc - 1] - d_compactMinValue[loc - 2];
        //real_t down = (real_t)(d_compactMinIndex[loc - 1] - d_compactMinIndex[loc - 2]);
        //slope_min = up / (real_t)down;
        slope_min = (d_compactMinValue[loc - 1] - d_compactMinValue[loc - 2]) / (d_compactMinIndex[loc - 1] - d_compactMinIndex[loc - 2]);
        t_min = d_compactMinValue[loc - 1] + slope_min * (d_compactMinIndex[loc] - d_compactMinIndex[loc - 1]);
        if (isinf(t_min))
            printf("inf error-4 in select extrema min \n");

        if (t_min < d_compactMinValue[loc])
        {
            //if (isinf(t_min))
            //{
            //	printf("inf error-3 in select extrema\n");
            //	printf("d_compactMinIndex[loc - 1]: %d \nd_compactMinIndex[loc - 2]: %d \n", d_compactMinIndex[loc - 1], d_compactMinIndex[loc - 2]);
            //	printf("down: %d\n", down);
            //	printf("slope_min: %f \n", slope_min);
            //}
            d_compactMinValue[loc] = t_min;
        }
    }
}

// for natural boundary conditions
template <typename coord_t, typename real_t>
__global__ void tridiagonal_setup(coord_t* d_num_extrema, coord_t* d_extrema_x, real_t* d_extrema_y, real_t* d_upper_dia, real_t* d_middle_dia, real_t* d_lower_dia, real_t* d_right_dia, size_t SignalLength, size_t SignalDim, size_t NumDirVector) {
    int dirVecIdx = blockIdx.z;
    int signalDimIdx = blockIdx.y;
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + pointsIdx;
    const int num_equation = d_num_extrema[dirVecIdx];
    int idxForRight = dirVecIdx * SignalDim * SignalLength + signalDimIdx * num_equation + pointsIdx; // to compact the y value within one direction vector
    if (pointsIdx == 0)
    {
        d_middle_dia[idx] = 1;
        d_upper_dia[idx] = 0;
        d_lower_dia[idx] = 0;
        d_right_dia[idxForRight] = 0; // it used to be idx
    }
    if (pointsIdx == num_equation - 1)
    {
        d_middle_dia[idx] = 1;
        d_lower_dia[idx] = 0;
        d_upper_dia[idx] = 0;
        d_right_dia[idxForRight] = 0;
    }
    if (pointsIdx != 0 && pointsIdx < num_equation - 1)
    {
        d_middle_dia[idx] = 2 * (((d_extrema_x[idx] - d_extrema_x[idx - 1]) + (d_extrema_x[idx + 1] - d_extrema_x[idx])));
        d_upper_dia[idx] = d_extrema_x[idx + 1] - d_extrema_x[idx];
        d_lower_dia[idx] = d_extrema_x[idx] - d_extrema_x[idx - 1];
        d_right_dia[idxForRight] = 3 * ((d_extrema_y[idx + 1] - d_extrema_y[idx]) / (d_extrema_x[idx + 1] - d_extrema_x[idx]) -
            (d_extrema_y[idx] - d_extrema_y[idx - 1]) / (d_extrema_x[idx] - d_extrema_x[idx - 1]));
    }
}

// for not-a-knot boundary conditions
template <typename coord_t, typename real_t>
__global__ void tridiagonal_setup_nak(coord_t* d_num_extrema, coord_t* d_extrema_x, real_t* d_extrema_y, real_t* d_upper_dia, real_t* d_middle_dia, real_t* d_lower_dia, real_t* d_right_dia, size_t SignalLength, size_t SignalDim, size_t NumDirVector) {
    int dirVecIdx = blockIdx.z;
    int signalDimIdx = blockIdx.y;
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + pointsIdx;
    const int num_equation = d_num_extrema[dirVecIdx] - 1;
    if (pointsIdx == 0)
    {
        real_t h0 = d_extrema_x[idx + 1] - d_extrema_x[idx];
        real_t h1 = d_extrema_x[idx + 2] - d_extrema_x[idx + 1];
        d_middle_dia[idx] = -h1 * h1 + h0 * h0;
        d_upper_dia[idx] = (h0 + h1) * h1 + 2 * h0 * (h0 + h1);
        d_lower_dia[idx] = 0; // fixed
        d_right_dia[idx] = 3 * (h0 / h1 * (d_extrema_y[idx + 2] - d_extrema_y[idx + 1]) - d_extrema_y[idx + 1] + d_extrema_y[idx]);
    }

    if (pointsIdx == num_equation)
    {
        real_t hn_2 = d_extrema_x[idx - 1] - d_extrema_x[idx - 2];
        real_t hn_1 = d_extrema_x[idx] - d_extrema_x[idx - 1];
        d_middle_dia[idx] = -hn_2 * hn_2 + hn_1 * hn_1;
        d_lower_dia[idx] = (hn_2 + hn_1) * hn_2 + hn_1 * 2 * (hn_2 + hn_1);
        d_upper_dia[idx] = 0; // fixed
        d_right_dia[idx] = 3 * ((d_extrema_y[idx] - d_extrema_y[idx - 1]) - hn_1 / hn_2 * (d_extrema_y[idx - 1] - d_extrema_y[idx - 2]));
    }
    if (pointsIdx != 0 && pointsIdx < num_equation)
    {
        d_middle_dia[idx] = 2 * (((d_extrema_x[idx] - d_extrema_x[idx - 1]) + (d_extrema_x[idx + 1] - d_extrema_x[idx])));
        d_upper_dia[idx] = d_extrema_x[idx + 1] - d_extrema_x[idx];
        d_lower_dia[idx] = d_extrema_x[idx] - d_extrema_x[idx - 1];
        d_right_dia[idx] = 3 * (d_extrema_y[idx + 1] - d_extrema_y[idx]) / (d_extrema_x[idx + 1] - d_extrema_x[idx]) - 3 * (d_extrema_y[idx] - d_extrema_y[idx - 1]) / (d_extrema_x[idx] - d_extrema_x[idx - 1]);
    }
}

template <typename coord_t, typename real_t>
__global__ void spline_coefficients(const real_t* a, real_t* b, real_t* c, real_t* d, coord_t* extrema_points_x, size_t SignalDim, size_t SignalLength, coord_t* d_num_extrema, real_t* solution) {
    int dirVecIdx = blockIdx.z;
    int signalDimIdx = blockIdx.y;
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + pointsIdx;

    const int num_equation = d_num_extrema[dirVecIdx];
    int idxForSolution = dirVecIdx * SignalDim * SignalLength + signalDimIdx * num_equation + pointsIdx; // to index the compacted sulution matrix, it used to be idx

    if (pointsIdx < num_equation - 1)
    {
        real_t h = extrema_points_x[idx + 1] - extrema_points_x[idx];
        b[idx] = (a[idx + 1] - a[idx]) / h - h * (2 * solution[idxForSolution] + solution[idxForSolution + 1]) / 3;
        d[idx] = (solution[idxForSolution + 1] - solution[idxForSolution]) / (3 * h);
        c[idx] = solution[idxForSolution];
    }

}

template <typename coord_t, typename real_t>
__global__ void interpolate(const real_t* a, real_t* b, real_t* c, real_t* d, coord_t* d_envelopeIndex, real_t* d_envelopeValue, coord_t* d_extremaIndex, size_t SignalLength, coord_t* d_num_extrema, size_t SignalDim, size_t NumDirVector) {
    int dirVecIdx = blockIdx.z;
    int signalDimIdx = blockIdx.y;
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + pointsIdx;
    int idx2 = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength;

    int num_coefs = d_num_extrema[dirVecIdx];
    size_t num_samples = SignalLength;
    if (pointsIdx < num_samples)// what the hell here? <= or <
    {
        int i = 0;
        int coef_idx = 0;
        int low = 1;
        int high = num_coefs - 1;
        // binary search for coef index
        while (low <= high) {
            int mid = (low + high) / 2;
            if ((pointsIdx > d_extremaIndex[idx2 + mid - 1]) && (pointsIdx <= d_extremaIndex[idx2 + mid])) {
                coef_idx = mid - 1;
                break;
            }
            else if (pointsIdx < d_extremaIndex[idx2 + mid]) {
                high = mid - 1;
            }
            else {
                low = mid + 1;
            }
        }

        coord_t t = d_envelopeIndex[idx] - d_extremaIndex[idx2 + coef_idx];
        d_envelopeValue[idx] = a[idx2 + coef_idx] + (b[idx2 + coef_idx] + (c[idx2 + coef_idx] + d[idx2 + coef_idx] * t) * t) * t;
    }

}

template <typename coord_t, typename real_t>
__global__ void averageUppperLower(real_t* d_meanEnvelope, real_t* d_upperEnvelope, real_t* d_lowerEnvelope, size_t SignalLength, size_t SignalDim, size_t NumDirVector, coord_t* d_num_extrema_max, coord_t* d_num_extrema_min)
{
    int dirVecIdx = blockIdx.z;
    int signalDimIdx = blockIdx.y;
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + pointsIdx;

    if ((d_num_extrema_max[dirVecIdx] > 3) && (d_num_extrema_min[dirVecIdx] > 3) && (pointsIdx < SignalLength))
    {
        d_meanEnvelope[idx] = (d_upperEnvelope[idx] + d_lowerEnvelope[idx]) / 2.0;
    }
}

template <typename coord_t, typename real_t>
__global__ void averageDirection(real_t* d_current, real_t* d_meanEnvelope, coord_t* d_num_extrema_max, coord_t* d_num_extrema_min, size_t NumDirVector, size_t SignalDim, size_t SignalLength)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SignalDim * SignalLength)
    {
        real_t value = 0;
        real_t numCompDir = 0;
        for (int i = 0; i < NumDirVector; i++)
        {
            if ((d_num_extrema_max[i] > 3) && (d_num_extrema_min[i] > 3))
            {
                value = value + d_meanEnvelope[i * SignalDim * SignalLength + idx];
                numCompDir = numCompDir + 1;
            }
        }
        d_current[idx] = d_current[idx] - value / numCompDir;
    }
}

template <typename real_t>
__global__ void updateSignal(real_t* d_current, real_t* d_running, size_t SignalDim, size_t SignalLength)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (SignalDim * SignalLength))
    {
        d_running[idx] = d_running[idx] - d_current[idx];
    }
}

__global__ void MatrixMulCUDA(float* C, float* A, float* B, int hA, int wA, int hB, int wB) {
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x; //< 30504
    int rowIdx = blockIdx.y;
    int idx = rowIdx * hA + colIdx;

    if (colIdx < hA)
    {
        double result = 0;
        double tem = 0;

        for (int i = 0; i < wA; i++)
        {
            tem = A[colIdx + i * hA] * B[rowIdx * hB + i];
            result = result + tem;
        }
        C[idx] = (float)result;
    }

}

int getBinSize(char* path)
{
    int  size = 0;
    FILE* fp = fopen(path, "rb");
    if (fp)
    {
        fseek(fp, 0, SEEK_END);
        size = ftell(fp);
        fclose(fp);
    }
    //printf("\npath=%s,size=%d \n", path, size);
    return size;
}

void readBin(char* path, char* buf, int size)
{
    FILE* infile;
    if ((infile = fopen(path, "rb")) == NULL)
    {
        printf("\nCan not open the path: %s \n", path);
        exit(-1);
    }
    fread(buf, sizeof(char), size, infile);
    fclose(infile);
}

void writeBin(char* path, char* buf, int size)
{
    FILE* outfile;
    if ((outfile = fopen(path, "wb")) == NULL)
    {
        printf("\nCan not open the path: %s \n", path);
        exit(-1);
    }
    fwrite(buf, sizeof(char), size, outfile);
    fclose(outfile);
}

__device__ float RadicalInverse(int Base, unsigned i)
{
    float Digit, Radical, Inverse;
    Digit = Radical = 1.0 / Base;
    Inverse = 0.0;
    while (i)
    {
        Inverse += Digit * (i % Base);
        Digit *= Radical;
        i /= Base;
    }
    return Inverse;
}

template <typename T>
__device__ float Sequence2Vector(T* sequence, T* vector, size_t dim, T* tht)
{
    // threads should load data from shared memory, instead of global memory
    int i = 0, j = 0;
    for (i = 0; i < dim; i++)
    {
        sequence[i] = 2 * sequence[i] - 1;
    }
    //T* tht = (T*)malloc((dim - 1) * sizeof(T)); // inefficient here, to be improved
    //T tht[dim - 1];
    for (i = 0; i < dim - 1; i++)
    {
        T tem = 0.0;
        for (j = dim - 1; j > i; j--)
        {
            tem = tem + pow(sequence[j], 2);
        }

        tht[i] = atan2(sqrt(tem), sequence[i]);
    }
    for (i = 0; i < dim; i++)
    {
        if (i == 0)
        {
            vector[i] = 1;
        }
        else
        {
            T tem = 1.0;
            for (j = 0; j < i; j++)
            {
                tem = tem * sin(tht[j]);
            }
            vector[i] = tem;
        }

    }
    for (i = 0; i < dim - 1; i++)
    {
        vector[i] = vector[i] * cos(tht[i]);
    }
}

template <typename T>
__global__ void GenerateHammSeq(int* d_PrimeDataSet, T* d_HammSeq)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x == 0)
    {
        d_HammSeq[gridDim.x * threadIdx.x + blockIdx.x] = (threadIdx.x + 1.5) / (-d_PrimeDataSet[blockIdx.x]);
    }
    else
    {
        d_HammSeq[gridDim.x * threadIdx.x + blockIdx.x] = RadicalInverse(d_PrimeDataSet[blockIdx.x], threadIdx.x + 1);
    }

}

template <typename T>
__global__ void GenerateDirVec(T* d_HammSeq, T* d_DirectionVector, size_t* d_SignalDim, T* d_tht)
{
    size_t dim = *d_SignalDim;
    Sequence2Vector(&d_HammSeq[dim * threadIdx.x], &d_DirectionVector[dim * threadIdx.x], dim, &d_tht[(dim - 1) * threadIdx.x]);
}

int isPrimes(int n)
{
    int i;
    int flag = 0;
    for (i = 2; i <= n / 2; ++i)
    {
        if (n % i == 0)
        {
            flag = 1;
            break;
        }
    }
    return flag;
}

template <typename T>
void GenerateDirectionVector(size_t SignalDim, size_t NumDirVector, T* d_DirectionVector)
{
    // generate primes dataset on cpu
    size_t NumPrimes = SignalDim;
    size_t PrimesCounter = 1;
    int PotentialPrimes = 2;
    int* h_PrimeDataSet = (int*)malloc(NumPrimes * sizeof(int));
    while (PrimesCounter < NumPrimes)
    {
        if (!isPrimes(PotentialPrimes))
        {
            h_PrimeDataSet[PrimesCounter] = PotentialPrimes;
            PrimesCounter++;
        }
        PotentialPrimes++;
    }
    h_PrimeDataSet[0] = -(int)NumDirVector;

    // generate hammersley sequence using primes dataset
    int* d_PrimeDataSet = NULL;
    T* d_HammSeq = NULL;
    cudaMalloc((void**)&d_HammSeq, SignalDim * NumDirVector * sizeof(T));
    cudaMalloc((void**)&d_PrimeDataSet, NumPrimes * sizeof(int));
    cudaMemcpy(d_PrimeDataSet, h_PrimeDataSet, NumPrimes * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block_dim = NumDirVector;
    dim3 grid_dim = SignalDim;
    GenerateHammSeq << <grid_dim, block_dim >> > (d_PrimeDataSet, d_HammSeq);

    // convert hammersley sequence to direction vector
    size_t* d_SignalDim = NULL;
    cudaMalloc((void**)&d_SignalDim, sizeof(size_t));
    cudaMemcpy(d_SignalDim, &SignalDim, sizeof(size_t), cudaMemcpyHostToDevice);

    T* d_tht = NULL;
    cudaMalloc((void**)&d_tht, (SignalDim - 1) * NumDirVector * sizeof(T));
    cudaMemset(d_tht, 0, (SignalDim - 1) * NumDirVector * sizeof(T));

    block_dim = NumDirVector;
    grid_dim = 1;
    GenerateDirVec << <grid_dim, block_dim >> > (d_HammSeq, d_DirectionVector, d_SignalDim, d_tht);

}

template <typename coord_t, typename real_t>
double memd(size_t SignalDim, size_t SignalLength, size_t NumDirVector, size_t num_IMFs, size_t max_iter, coord_t* d_x, real_t* d_y, real_t* d_DirectionVectors, real_t* d_IMFs)
{
    //===============load data===============
    real_t* d_current = NULL;
    cudaMalloc((void**)&d_current, SignalDim * SignalLength * sizeof(real_t));
    real_t* d_running = NULL;
    cudaMalloc((void**)&d_running, SignalDim * SignalLength * sizeof(real_t));

    //===============preparation for projection===============
    real_t* d_ProjectSignals = NULL;
    cudaMalloc((void**)&d_ProjectSignals, NumDirVector * SignalLength * sizeof(real_t));
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    float alpha = 1;
    float beta = 0;

    //===============preparation for extreme points detection===============
    coord_t* d_sparseFlag;
    cudaMalloc((void**)&d_sparseFlag, NumDirVector * SignalLength * sizeof(coord_t));

    coord_t* d_multiProjectSignalIndex = NULL;
    cudaMalloc((void**)&d_multiProjectSignalIndex, NumDirVector * SignalLength * sizeof(coord_t));
    for (int i = 0; i < NumDirVector; i++)
    {
        int head = i * SignalLength;
        cudaMemcpy((d_multiProjectSignalIndex + head), d_x, SignalLength * sizeof(coord_t), cudaMemcpyDeviceToDevice);
    }

    //===============preparation for prefix sum===============
    coord_t* d_ScanResult;
    cudaMalloc((void**)&d_ScanResult, NumDirVector * SignalLength * sizeof(coord_t));
    int* d_sums, * d_incr;
    cudaMalloc((void**)&d_sums, SignalLength * sizeof(int));
    cudaMalloc((void**)&d_incr, SignalLength * sizeof(int));

    //===============preparation for extreme points select===============
    real_t* d_compactValue;
    coord_t* d_compactIndex;
    coord_t* d_num_extrema_max, * d_num_extrema_min;

    cudaMalloc((void**)&d_compactValue, NumDirVector * SignalDim * SignalLength * sizeof(real_t));
    cudaMalloc((void**)&d_compactIndex, NumDirVector * SignalDim * SignalLength * sizeof(coord_t));
    cudaMalloc((void**)&d_num_extrema_max, NumDirVector * sizeof(coord_t));
    cudaMalloc((void**)&d_num_extrema_min, NumDirVector * sizeof(coord_t));

    //===============preparation for tridiagonal setting===============
    real_t* d_upperDia = NULL, * d_middleDia = NULL, * d_lowerDia = NULL, * d_right = NULL;

    cudaMalloc((void**)&d_upperDia, NumDirVector * SignalDim * SignalLength * sizeof(real_t));
    cudaMalloc((void**)&d_middleDia, NumDirVector * SignalDim * SignalLength * sizeof(real_t));
    cudaMalloc((void**)&d_lowerDia, NumDirVector * SignalDim * SignalLength * sizeof(real_t));
    cudaMalloc((void**)&d_right, NumDirVector * SignalDim * SignalLength * sizeof(real_t));

    //===============preparation for tridiagonal solving===============
    real_t* d_solutionGtsv = NULL;
    cudaMalloc((void**)&d_solutionGtsv, NumDirVector * SignalDim * SignalLength * sizeof(real_t));

    real_t* currentUpperDia = NULL;
    real_t* currentMiddleDia = NULL;
    real_t* currentLowerDia = NULL;
    real_t* currentRightDia = NULL;
    real_t* currentSolution = NULL;
    coord_t* h_num_extrema = (coord_t*)malloc(NumDirVector * sizeof(coord_t));

    cusparseHandle_t handle_sparse;
    cusparseCreate(&handle_sparse);
    size_t* buffer_size = (size_t*)malloc(sizeof(size_t));
    real_t* buffer = NULL;
    cudaMalloc(&buffer, 1024 * 1024 * sizeof(real_t));

    //===============preparation for spline coefficients calculation===============
    real_t* d_b = NULL, * d_d = NULL, * d_c = NULL;

    //===============preparation for interpolate values===============
    coord_t* d_multiDirVecChanSignalIndex = NULL;
    cudaMalloc((void**)&d_multiDirVecChanSignalIndex, NumDirVector * SignalDim * SignalLength * sizeof(coord_t));

    for (int i = 0; i < NumDirVector * SignalDim; i++)
    {
        int head = i * SignalLength;
        cudaMemcpy((d_multiDirVecChanSignalIndex + head), d_x, SignalLength * sizeof(coord_t), cudaMemcpyDeviceToDevice);
    }

    real_t* d_envelopeVauleMax = NULL, * d_envelopeVauleMin = NULL;
    cudaMalloc((void**)&d_envelopeVauleMax, NumDirVector * SignalDim * SignalLength * sizeof(real_t));
    cudaMalloc((void**)&d_envelopeVauleMin, NumDirVector * SignalDim * SignalLength * sizeof(real_t));

    //===============preparation for averaging upper and lower===============
    real_t* d_meanEnvelope = NULL;
    cudaMalloc((void**)&d_meanEnvelope, NumDirVector * SignalDim * SignalLength * sizeof(real_t));

    //===============preparation for MULTI-STREAM===============
    //const int num_streams = 2;
    //cudaStream_t streams[num_streams];
    //cudaStreamCreate(&streams[1]);
    //cudaStreamCreate(&streams[2]);

    //===============start looping===============
    cudaMemcpy(d_running, d_y, SignalDim * SignalLength * sizeof(real_t), cudaMemcpyDeviceToDevice);
    double start_time = omp_get_wtime();
    for (size_t i = 0; i < num_IMFs - 1; ++i)
    {
        cudaMemcpy(d_current, d_running, SignalDim * SignalLength * sizeof(real_t), cudaMemcpyDeviceToDevice);
        for (size_t j = 0; j < max_iter; ++j)
        {
            //==================multivariate signal projection============
            cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, SignalLength, NumDirVector, SignalDim,
                &alpha, d_current, SignalLength, d_DirectionVectors, SignalDim,
                &beta, d_ProjectSignals, SignalLength);

            //dim3 threads(1024);
            //dim3 grid(SignalLength / threads.x + 1, NumDirVector);
            //MatrixMulCUDA <<<grid, threads >>> (d_ProjectSignals, d_current, d_DirectionVectors, SignalLength, SignalDim, SignalDim, NumDirVector);

            //==================extreme points detection max============
            dim3 blockDimShfl(256);
            size_t offset = (blockDimShfl.x / 32 - 1) * 2;
            dim3 gridDimShfl(SignalLength / (blockDimShfl.x - offset) + (SignalLength % (blockDimShfl.x - offset) == 0 ? 0 : 1), NumDirVector);
            cudaMemset(d_sparseFlag, 0, NumDirVector * SignalLength * sizeof(coord_t));
            find_extrema_shfl_max << <gridDimShfl, blockDimShfl >> > (d_multiProjectSignalIndex, d_ProjectSignals, d_sparseFlag, SignalLength);

            //==================prefix scan max============
            for (size_t k = 0; k < NumDirVector; k++)
            {
                int offset = k * SignalLength;
                if (SignalLength > ELEMENTS_PER_BLOCK) {
                    scanLargeDeviceArray(d_ScanResult + offset, d_sparseFlag + offset, SignalLength, 1, d_sums, d_incr);
                }
                else {
                    scanSmallDeviceArray(d_ScanResult + offset, d_sparseFlag + offset, SignalLength, 1);
                }
            }

            //==================extreme points select max============
            dim3 blockDimSelectExtrema(256);
            dim3 gridDimSelectExtrema(SignalLength / blockDimSelectExtrema.x + (SignalLength % blockDimSelectExtrema.x == 0 ? 0 : 1), SignalDim, NumDirVector);
            select_extrema_max << <gridDimSelectExtrema, blockDimSelectExtrema >> > (d_sparseFlag, d_current, d_multiDirVecChanSignalIndex,
                d_ScanResult, d_compactValue, d_compactIndex, SignalLength, SignalDim, NumDirVector, d_num_extrema_max);

            dim3 blockDimSetBoundary(2);
            dim3 gridDimSetBoundary(1, SignalDim, NumDirVector);
            setBoundaryMax << <gridDimSetBoundary, blockDimSetBoundary >> > (d_compactValue, d_compactIndex,
                d_ScanResult, SignalLength, SignalDim, NumDirVector);

            //==================set up tridiagonal matrix max============
            dim3 blockDimTriSet(256);
            dim3 gridDimTriSet(SignalLength / blockDimTriSet.x + (SignalLength % blockDimTriSet.x == 0 ? 0 : 1), SignalDim, NumDirVector); // too much idle threads
            tridiagonal_setup << <gridDimTriSet, blockDimTriSet >> > (d_num_extrema_max, d_compactIndex, d_compactValue,
                d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength, SignalDim, NumDirVector);

            //==================solve tridiagonal matrix max============
            cudaMemcpy(d_solutionGtsv, d_right, NumDirVector * SignalDim * SignalLength * sizeof(real_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(h_num_extrema, d_num_extrema_max, NumDirVector * sizeof(coord_t), cudaMemcpyDeviceToHost);
            for (size_t k = 0; k < NumDirVector; k++)
            {
                currentUpperDia = d_upperDia + k * SignalLength * SignalDim;
                currentMiddleDia = d_middleDia + k * SignalLength * SignalDim;
                currentLowerDia = d_lowerDia + k * SignalLength * SignalDim;
                currentRightDia = d_right + k * SignalLength * SignalDim;
                currentSolution = d_solutionGtsv + k * SignalLength * SignalDim;

                cusparseSgtsv2_nopivot_bufferSizeExt(handle_sparse, h_num_extrema[k], SignalDim, currentLowerDia, currentMiddleDia, currentUpperDia, currentRightDia, h_num_extrema[k], buffer_size);
                //cudaMalloc(&buffer, *buffer_size); //to be optimized
                cusparseSgtsv2_nopivot(handle_sparse, h_num_extrema[k], SignalDim, currentLowerDia, currentMiddleDia, currentUpperDia, currentSolution, h_num_extrema[k], buffer);
                //cudaFree(buffer);
            }

            //==================compute spline coefficients max============
            dim3 blockDimSplineCoe(256);
            dim3 gridDimSplineCoe(SignalLength / blockDimSplineCoe.x + (SignalLength % blockDimSplineCoe.x == 0 ? 0 : 1), SignalDim, NumDirVector); // too much idle threads
            //cudaMemset(d_upperDia, 0, NumDirVector * SignalDim * SignalLength * sizeof(real_t)); // to replace d_b
            //cudaMemset(d_middleDia, 0, NumDirVector * SignalDim * SignalLength * sizeof(real_t)); // to replace d_c
            //cudaMemset(d_lowerDia, 0, NumDirVector * SignalDim * SignalLength * sizeof(real_t)); // to replace d_d
            spline_coefficients << <gridDimSplineCoe, blockDimSplineCoe >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_compactIndex, SignalDim, SignalLength, d_num_extrema_max, d_solutionGtsv);

            //==================interpolate values max============
            dim3 blockDimInterpolate(256);
            dim3 gridDimInterpolate(SignalLength / blockDimInterpolate.x + (SignalLength % blockDimInterpolate.x == 0 ? 0 : 1), SignalDim, NumDirVector);
            interpolate << <gridDimInterpolate, blockDimInterpolate >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_multiDirVecChanSignalIndex, d_envelopeVauleMax, d_compactIndex, SignalLength, d_num_extrema_max, SignalDim, NumDirVector);

            //==================extreme points detection min============
            cudaMemset(d_sparseFlag, 0, NumDirVector * SignalLength * sizeof(coord_t));
            find_extrema_shfl_min << <gridDimShfl, blockDimShfl >> > (d_multiProjectSignalIndex, d_ProjectSignals, d_sparseFlag, SignalLength);

            //==================prefix scan min============
            for (size_t k = 0; k < NumDirVector; k++)
            {
                int offset = k * SignalLength;
                if (SignalLength > ELEMENTS_PER_BLOCK) {
                    scanLargeDeviceArray(d_ScanResult + offset, d_sparseFlag + offset, SignalLength, 1, d_sums, d_incr);
                }
                else {
                    scanSmallDeviceArray(d_ScanResult + offset, d_sparseFlag + offset, SignalLength, 1);
                }
            }

            //==================extreme points select min============
            select_extrema_min << <gridDimSelectExtrema, blockDimSelectExtrema >> > (d_sparseFlag, d_current, d_multiDirVecChanSignalIndex,
                d_ScanResult, d_compactValue, d_compactIndex, SignalLength, SignalDim, NumDirVector, d_num_extrema_min);

            setBoundaryMin << <gridDimSetBoundary, blockDimSetBoundary >> > (d_compactValue, d_compactIndex,
                d_ScanResult, SignalLength, SignalDim, NumDirVector);

            //==================set up tridiagonal matrix min============
            tridiagonal_setup << <gridDimTriSet, blockDimTriSet >> > (d_num_extrema_min, d_compactIndex, d_compactValue,
                d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength, SignalDim, NumDirVector);

            //==================solve tridiagonal matrix min============
            cudaMemcpy(d_solutionGtsv, d_right, NumDirVector * SignalDim * SignalLength * sizeof(real_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(h_num_extrema, d_num_extrema_min, NumDirVector * sizeof(coord_t), cudaMemcpyDeviceToHost);

            for (size_t k = 0; k < NumDirVector; k++)
            {
                currentUpperDia = d_upperDia + k * SignalLength * SignalDim;
                currentMiddleDia = d_middleDia + k * SignalLength * SignalDim;
                currentLowerDia = d_lowerDia + k * SignalLength * SignalDim;
                currentRightDia = d_right + k * SignalLength * SignalDim;
                currentSolution = d_solutionGtsv + k * SignalLength * SignalDim;

                cusparseSgtsv2_nopivot_bufferSizeExt(handle_sparse, h_num_extrema[k], SignalDim, currentLowerDia, currentMiddleDia, currentUpperDia, currentRightDia, h_num_extrema[k], buffer_size);
                //cudaMalloc(&buffer, *buffer_size); //to be optimized
                cusparseSgtsv2_nopivot(handle_sparse, h_num_extrema[k], SignalDim, currentLowerDia, currentMiddleDia, currentUpperDia, currentSolution, h_num_extrema[k], buffer);
                //cudaFree(buffer);
            }

            //==================compute spline coefficients min============
            //cudaMemset(d_upperDia, 0, NumDirVector* SignalDim* SignalLength * sizeof(real_t)); // to replace d_b
            //cudaMemset(d_middleDia, 0, NumDirVector* SignalDim* SignalLength * sizeof(real_t)); // to replace d_c
            //cudaMemset(d_lowerDia, 0, NumDirVector* SignalDim* SignalLength * sizeof(real_t)); // to replace d_d
            spline_coefficients << <gridDimSplineCoe, blockDimSplineCoe >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_compactIndex, SignalDim, SignalLength, d_num_extrema_min, d_solutionGtsv);

            //==================interpolate values min============
            interpolate << <gridDimInterpolate, blockDimInterpolate >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_multiDirVecChanSignalIndex, d_envelopeVauleMin, d_compactIndex, SignalLength, d_num_extrema_min, SignalDim, NumDirVector);

            //==================average upper and lower============
            dim3 blockDimMeanEnvelope(256);
            dim3 gridDimMeanEnvelope(SignalLength / blockDimInterpolate.x + (SignalLength % blockDimInterpolate.x == 0 ? 0 : 1), SignalDim, NumDirVector);
            averageUppperLower << <gridDimMeanEnvelope, blockDimMeanEnvelope >> > (d_meanEnvelope, d_envelopeVauleMax, d_envelopeVauleMin, SignalLength, SignalDim, NumDirVector, d_num_extrema_max, d_num_extrema_min);

            //==================average each direction and update d_current============
            dim3 blockDimMeanDir(256);
            dim3 gridDimMeanDir(SignalDim * SignalLength / blockDimMeanDir.x + (SignalDim * SignalLength % blockDimMeanDir.x == 0 ? 0 : 1));
            averageDirection << <gridDimMeanDir, blockDimMeanDir >> > (d_current, d_meanEnvelope, d_num_extrema_max, d_num_extrema_min, NumDirVector, SignalDim, SignalLength);

            //=============================TEST============================
            //real_t* h_current_new = (real_t*)malloc(SignalDim * SignalLength * sizeof(real_t));
            //cudaMemcpy(h_current_new, d_current, SignalDim * SignalLength * sizeof(real_t), cudaMemcpyDeviceToHost);
            //char test_file[] = "C:\\Users\\BRAIN-1\\source\\repos\\EMD project\\MEMD_Algorithm\\MEMD\\test_1.bin";
            //writeBin(test_file, (char*)h_test, SignalDim * SignalLength * sizeof(real_t));
            //printf("IMF: %d, Iter: %d, averageDir %f \n", i, j, h_current_new[0]);
            //=============================TEST============================
        }
        cudaMemcpy(&d_IMFs[i * SignalDim * SignalLength], d_current, SignalDim * SignalLength * sizeof(real_t), cudaMemcpyDeviceToDevice);

        dim3 blockDimUpdateSignal(256);
        dim3 gridDimUpdateSignal(SignalDim * SignalLength / blockDimUpdateSignal.x + (SignalDim * SignalLength % blockDimUpdateSignal.x == 0 ? 0 : 1));
        updateSignal << <gridDimUpdateSignal, blockDimUpdateSignal >> > (d_current, d_running, SignalDim, SignalLength);
        //printf("%s, IMF: %d, updateSignal \n", cudaGetErrorString(cudaGetLastError()), i);

    }
    double end_time = omp_get_wtime();
    double diff = end_time - start_time;
    cudaMemcpy(&d_IMFs[(num_IMFs - 1) * SignalDim * SignalLength], d_running, SignalDim * SignalLength * sizeof(real_t), cudaMemcpyDeviceToDevice);

    //free all the CPU and GPU memory here
    cudaFree(d_current);
    cudaFree(d_running);
    cudaFree(d_ProjectSignals);
    cudaFree(d_sparseFlag);
    cudaFree(d_multiProjectSignalIndex);
    cudaFree(d_ScanResult);
    cudaFree(d_sums);
    cudaFree(d_incr);
    cudaFree(d_compactValue);
    cudaFree(d_compactIndex);
    cudaFree(d_num_extrema_max);
    cudaFree(d_num_extrema_min);
    cudaFree(d_upperDia);
    cudaFree(d_middleDia);
    cudaFree(d_lowerDia);
    cudaFree(d_right);
    cudaFree(d_solutionGtsv);
    //cudaFree(d_b);
    //cudaFree(d_d);
    //cudaFree(d_c);
    cudaFree(d_multiDirVecChanSignalIndex);
    cudaFree(d_envelopeVauleMax);
    cudaFree(d_envelopeVauleMin);
    cudaFree(d_meanEnvelope);
    free(h_num_extrema);
    cudaFree(buffer);
    return diff;
}

void runBenchmark()
{
    size_t num_IMFs = 8;
    size_t max_iter = 10;

    //char filePath1[] = "./S6.bin";
    //char filePath2[] = "./S16.bin";
    char filePath1[] = "./InputData_32CH_244032.bin";
    char filePath2[] = "./InputData_64CH_244032.bin";
    char filePath3[] = "./InputData_128CH_244032.bin";
    char* allFilePath[] = { filePath1, filePath2, filePath3 };
    //char* allFilePath[] = { filePath1, filePath2 };
    int signalDimension[3] = { 32, 64, 128 };
    int directionVectors[3] = { 64, 128, 256 };
    //int signalLength[1] = { 2049 };
    int signalLength[118] = { 2049, 4097, 6145, 8193, 10241, 12289, 14337, 16385, 18433, 20481,
                            22529, 24577, 26625, 28673, 30721, 32769, 34817, 36865, 38913, 40961,
                            43009, 45057, 47105, 49153,	51201, 53249, 55297, 57345,	59393, 61441,
                            63489, 65537, 67585, 69633,	71681, 73729, 75777, 77825,	79873, 81921,
                            83969, 86017, 88065, 90113,	92161, 94209, 96257, 98305,	100353, 102401,
                            104449,	106497,	108545, 110593, 112641, 114689,	116737,	118785,	120833,
                            122881,	124929,	126977,	129025,	131073,	133121,	135169,	137217,	139265,
                            141313,	143361, 145409, 147457, 149505,	151553,	153601,	155649,	157697,
                            159745,	161793,	163841,	165889,	167937,	169985,	172033,	174081, 176129,
                            178177,	180225,	182273,	184321,	186369,	188417,	190465,	192513,	194561,
                            196609,	198657,	200705,	202753,	204801,	206849,	208897,	210945,	212993,
                            215041,	217089,	219137, 221185,	223233,	225281,	227329,	229377,	231425,
                            233473,	235521,	237569,	239617,	241665 };

    for (int fileIdx = 1; fileIdx < 2; fileIdx++) // iterate channel: 32, 64, 128
    {
        char* filePath = allFilePath[fileIdx];
        size_t nbytes = getBinSize(filePath); // in bytes
        const size_t SignalLength = nbytes / sizeof(float) / signalDimension[fileIdx];
        size_t oneChannelNbytes_coord = SignalLength * sizeof(int); // one dimention of the multivariate input signal
        size_t oneChannelNbytes_real = SignalLength * sizeof(float);

        // allocate array on host
        float* h_y;
        int* h_x;
        h_x = (int*)malloc(oneChannelNbytes_coord);
        h_y = (float*)malloc(oneChannelNbytes_real * signalDimension[fileIdx]);
        // load data
        char* buf = (char*)malloc(oneChannelNbytes_real * signalDimension[fileIdx]);
        readBin(filePath, buf, oneChannelNbytes_real * signalDimension[fileIdx]);
        h_y = (float*)buf;
        // generate data index
        for (int i = 0; i < SignalLength; i++) {
            h_x[i] = i;
        }

        for (int dirVecIdx = 1; dirVecIdx < 2; dirVecIdx++) // iterate direction vectors: 64, 128, 256
        {
            for (int sigLenIdx = 2; sigLenIdx < 3; sigLenIdx++) // iterate signal length
            {
                float* d_y;
                int* d_x;
                size_t oneChannelNbytes_coord = signalLength[sigLenIdx] * sizeof(int); // one dimention of the multivariate input signal
                size_t oneChannelNbytes_real = signalLength[sigLenIdx] * sizeof(float);
                // allocate array on device
                cudaMalloc((void**)&d_x, oneChannelNbytes_coord);
                cudaMalloc((void**)&d_y, oneChannelNbytes_real * signalDimension[fileIdx]);
                // copy data to device
                cudaMemcpy(d_x, h_x, oneChannelNbytes_coord, cudaMemcpyHostToDevice);
                for (int sigDimIdx = 0; sigDimIdx < signalDimension[fileIdx]; sigDimIdx++)
                {
                    float* current_h_y = h_y + sigDimIdx * SignalLength;
                    float* current_d_y = d_y + sigDimIdx * signalLength[sigLenIdx];
                    cudaMemcpy(current_d_y, current_h_y, oneChannelNbytes_real, cudaMemcpyHostToDevice);
                }

                // direction vector generation
                float* d_DirectionVectors = NULL;
                cudaMalloc(&d_DirectionVectors, signalDimension[fileIdx] * directionVectors[dirVecIdx] * sizeof(float));
                GenerateDirectionVector(signalDimension[fileIdx], directionVectors[dirVecIdx], d_DirectionVectors);

                float* d_IMFs = NULL;
                cudaMalloc(&d_IMFs, num_IMFs * signalDimension[fileIdx] * signalLength[sigLenIdx] * sizeof(float));// each IMF is a multivariate IMF
                double time = memd(signalDimension[fileIdx], signalLength[sigLenIdx], directionVectors[dirVecIdx], num_IMFs, max_iter, d_x, d_y, d_DirectionVectors, d_IMFs);
                printf("Channel: %d, Length: %d, DirVec: %d, Time: %f \n", signalDimension[fileIdx], signalLength[sigLenIdx], directionVectors[dirVecIdx], time);

                float* IMFs = (float*)malloc(num_IMFs * signalDimension[fileIdx] * signalLength[sigLenIdx] * sizeof(float));
                cudaMemcpy(IMFs, d_IMFs, num_IMFs * signalDimension[fileIdx] * signalLength[sigLenIdx] * sizeof(float), cudaMemcpyDeviceToHost);
                char IMFs_file[] = "./mode_matrix.bin";
                writeBin(IMFs_file, (char*)IMFs, num_IMFs * signalDimension[fileIdx] * signalLength[sigLenIdx] * sizeof(float));

                //free all the mem
                cudaFree(d_y);
                cudaFree(d_x);
                cudaFree(d_DirectionVectors);
                cudaFree(d_IMFs);
            }
        }
    }
}

int main()
{
    //runBenchmark();

    //=============================Single running=============================

    configuration for the input signal
    size_t SignalDim = 32;
    size_t NumDirVector = 64;
    size_t num_IMFs = 9;
    size_t max_iter = 10;

    float* h_y, * d_y;
    int* h_x, * d_x;

    // get data size
    //char* filePathInput;
    //filePathInput = argv[5];
    char filePathInput[] = "./EEGsample_32CH_30504.bin";
    size_t nbytes = getBinSize(filePathInput); // in bytes
    const size_t SignalLength = nbytes / sizeof(float) / SignalDim;
    size_t oneChannelNbytes_coord = SignalLength * sizeof(int); // one dimention of the multivariate input signal
    size_t oneChannelNbytes_real = SignalLength * sizeof(float);

    // allocate array on host
    h_x = (int*)malloc(oneChannelNbytes_coord);
    h_y = (float*)malloc(oneChannelNbytes_real * SignalDim);

    // allocate array on device
    cudaMalloc((void**)&d_x, oneChannelNbytes_coord);
    cudaMalloc((void**)&d_y, oneChannelNbytes_real * SignalDim);

    // load data
    char* buf = (char*)malloc(oneChannelNbytes_real * SignalDim);
    readBin(filePathInput, buf, oneChannelNbytes_real * SignalDim);
    h_y = (float*)buf;

    // generate data index
    for (size_t i = 0; i < SignalLength; i++) {
        h_x[i] = (int)i;
    }

    // copy data to device
    cudaMemcpy(d_x, h_x, oneChannelNbytes_coord, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, oneChannelNbytes_real * SignalDim, cudaMemcpyHostToDevice);

    // direction vector generation
    float* d_DirectionVectors = NULL;
    cudaMalloc(&d_DirectionVectors, SignalDim * NumDirVector * sizeof(float));
    GenerateDirectionVector(SignalDim, NumDirVector, d_DirectionVectors);

    //memd processing
    float* d_IMFs = NULL;
    cudaMalloc(&d_IMFs, num_IMFs * SignalDim * SignalLength * sizeof(float));// each IMF is a multivariate IMF
    float* IMFs = (float*)malloc(num_IMFs * SignalDim * SignalLength * sizeof(float));

    memd(SignalDim, SignalLength, NumDirVector, num_IMFs, max_iter, d_x, d_y, d_DirectionVectors, d_IMFs);

    cudaMemcpy(IMFs, d_IMFs, num_IMFs * SignalDim * SignalLength * sizeof(float), cudaMemcpyDeviceToHost);
    //char* IMFs_file = argv[6];
    char IMFs_file[] = "./mode_matrix.bin";
    writeBin(IMFs_file, (char*)IMFs, num_IMFs * SignalDim * SignalLength * sizeof(float));

}