#ifndef CUDAHELPER_HPP
#define CUDAHELPER_HPP


#include <helper_cuda.h>
#include "vector"
#include "checkCudaCall.hpp"

__device__ int atomicAdd(int* address, int val);
__device__ unsigned int atomicAdd(unsigned int* address,unsigned int val);
__device__ unsigned long long int atomicAdd(unsigned long long int* address,unsigned long long int val);
__device__ float atomicAdd(float* address, float val);
__device__ double atomicAdd(double* address, double val);
__device__ unsigned int atomicInc(unsigned int* address,unsigned int val);

__device__ int atomicAdd_block(int* address, int val);
__device__ unsigned int atomicAdd_block(unsigned int* address,unsigned int val);
__device__ unsigned long long int atomicAdd_block(unsigned long long int* address,unsigned long long int val);
__device__ float atomicAdd_block(float* address, float val);
__device__ double atomicAdd_block(double* address, double val);
__device__ unsigned int atomicInc_block(unsigned int* address,unsigned int val);


__device__ int atomicExch(int* address, int val);
__device__ unsigned int atomicExch(unsigned int* address,unsigned int val);
__device__ unsigned long long int atomicExch(unsigned long long int* address,unsigned long long int val);
__device__ float atomicExch(float* address, float val);

__device__ int atomicExch_block(int* address, int val);
__device__ unsigned int atomicExch_block(unsigned int* address,unsigned int val);
__device__ unsigned long long int atomicExch_block(unsigned long long int* address,unsigned long long int val);
__device__ float atomicExch_block(float* address, float val);

__device__ int atomicMax(int* address, int val);
__device__ unsigned int atomicMax(unsigned int* address,unsigned int val);
__device__ unsigned long long int atomicMax(unsigned long long int* address,unsigned long long int val);

__device__ int atomicMax_block(int* address, int val);
__device__ unsigned int atomicMax_block(unsigned int* address,unsigned int val);
__device__ unsigned long long int atomicMax_block(unsigned long long int* address,unsigned long long int val);


class CudaMemoryManager
{
public:
    static CudaMemoryManager& getInstance()
    {
        static CudaMemoryManager instance;
        return instance;
    }

    CudaMemoryManager(CudaMemoryManager const&) = delete;
    void operator= (CudaMemoryManager const&) = delete;

    void reset()
    {
        _bytes = 0;
    }

    template<typename T>
    void acquireMemory(uint64_t arraySize, T*& result)
    {
        checkCudaCall(cudaMalloc(&result, sizeof(T)*arraySize));
        _bytes += sizeof(T)*arraySize;
    }

    template<typename T>
    void freeMemory(T& memory)
    {
        checkCudaCall(cudaFree(memory));
    }

    uint64_t getSizeOfAcquiredMemory() const
    {
        return _bytes;
    }

private:
    CudaMemoryManager() {}
    ~CudaMemoryManager() {}

    uint64_t _bytes = 0;
};



__device__ inline float toFloat(int value)
{
    return static_cast<float>(value);
}

__device__ inline int toInt(float value)
{
    return static_cast<int>(value);
}



class CudaNumberGenerator
{
private:
    unsigned int* _currentIndex;
    int* _array;
    int _size;

    unsigned long long int* _currentId;

public:
    void init(int size)
    {
        _size = size;

        CudaMemoryManager::getInstance().acquireMemory<unsigned int>(1, _currentIndex);
        CudaMemoryManager::getInstance().acquireMemory<int>(size, _array);
        CudaMemoryManager::getInstance().acquireMemory<unsigned long long int>(1, _currentId);

        checkCudaCall(cudaMemset(_currentIndex, 0, sizeof(unsigned int)));
        unsigned long long int hostCurrentId = 1;
        checkCudaCall(cudaMemcpy(_currentId, &hostCurrentId, sizeof(_currentId), cudaMemcpyHostToDevice));

        std::vector<int> randomNumbers(size);
        for (int i = 0; i < size; ++i) {
            randomNumbers[i] = rand();
        }
        checkCudaCall(cudaMemcpy(_array, randomNumbers.data(), sizeof(int) * size, cudaMemcpyHostToDevice));
    }


    __device__ __inline__ int random(int maxVal)
    {
        int number = getRandomNumber();
        return number % (maxVal + 1);
    }

    __device__ __inline__ float random(float maxVal)
    {
        int number = getRandomNumber();
        return maxVal * static_cast<float>(number) / RAND_MAX;
    }

    __device__ __inline__ float random()
    {
        int number = getRandomNumber();
        return static_cast<float>(number) / RAND_MAX;
    }

    __device__ __inline__ unsigned long long int createNewId_kernel() { return atomicAdd(_currentId, 1); }

    void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_currentIndex);
        CudaMemoryManager::getInstance().freeMemory(_array);
        CudaMemoryManager::getInstance().freeMemory(_currentId);

        cudaFree(_currentId);
        cudaFree(_currentIndex);
        cudaFree(_array);
    }

private:
    __device__ __inline__ int getRandomNumber()
    {
        int index = atomicInc(_currentIndex, _size - 1);
        return _array[index];
    }
};

#endif // CUDASOLVER_HPP
