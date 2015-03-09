#ifndef SORT_INTERFACE_H
#define SORT_INTERFACE_H

#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types_common.h"
#include "cuda.h"


/*
Base class for sorting.

For testing purposes memory management methods are public.
*/
class Sort
{
protected:
    // Array of keys on host
    data_t *h_keys = NULL;
    // Length of array
    uint_t arrayLength = 0;
    // Sort order (ascending or descending)
    order_t sortOrder = ORDER_ASC;
    // Sort type
    sort_type_t sortType = (sort_type_t) NULL;

    /*
    Executes the sort.
    */
    void sortPrivate();

public:
    Sort(){};
    ~Sort()
    {
        memoryDestroy();
    }

    /*
    Method for allocating memory needed for sort.
    */
    void memoryAllocate(uint_t arrayLength) {}

    /*
    Method for destroying memory needed for sort.
    */
    void memoryDestroy() {}

    /*
    Provides a wrapper for private sort.
    */
    void sort(data_t *h_keys, uint_t arrayLength);
};


/*
Base class for sequential sort of keys only.
*/
class SortSequentialKeyOnly : Sort
{
    // Sequential sort for keys only
    sort_type_t sortType = SORT_SEQUENTIAL_KEY_ONLY;

    /*
    Provides a wrapper for private sort.
    */
    void sort(data_t *h_keys, uint_t arrayLength, order_t sortOrder)
    {
        if (arrayLength > this->arrayLength)
        {
            memoryDestroy();
            memoryAllocate(arrayLength);
        }

        this->h_keys = h_keys;
        this->arrayLength = arrayLength;
        this->sortOrder = sortOrder;

        sortPrivate();
    }
};


/*
Base class for sequential sort of key-value pairs.
*/
class SortSequentialKeyValue : public Sort
{
protected:
    // Array of values on host
    data_t *h_values = NULL;
    // Sequential sort for key-value pairs
    sort_type_t sortType = SORT_SEQUENTIAL_KEY_VALUE;

    /*
    Provides a wrapper for private sort.
    */
    void sort(data_t *h_keys, data_t *h_values, uint_t arrayLength, order_t sortOrder)
    {
        if (arrayLength > this->arrayLength)
        {
            memoryDestroy();
            memoryAllocate(arrayLength);
        }

        this->h_keys = h_keys;
        this->h_values = h_values;
        this->arrayLength = arrayLength;
        this->sortOrder = sortOrder;

        sortPrivate();
    }
};


/*
Base class for parallel sort of keys only.
*/
class SortParallelKeyOnly : public Sort
{
protected:
    // Array for keys on device
    data_t *d_keys = NULL;
    // Designates if data has been copied to device
    bool memoryCopiedToDevice = false;
    // Designates if data should be copied from device to host after the sort is completed
    bool memoryCopyAfterSort = true;
    // Parallel sort for keys only
    sort_type_t sortType = SORT_PARALLEL_KEY_ONLY;

public:
    /*
    Method for allocating memory needed for sort.
    */
    void memoryAllocate(uint_t arrayLength)
    {
        cudaError_t error = cudaMalloc((void **)this->d_keys, arrayLength * sizeof(*(this->d_keys)));
        checkCudaError(error);
    }

    /*
    Method for destroying memory needed for sort.
    */
    void memoryDestroy()
    {
        cudaError_t error = cudaFree(this->d_keys);
        checkCudaError(error);
    }

    /*
    Copies data from host to device
    */
    void memoryCopyHostToDevice(data_t *h_keys, uint_t arrayLength)
    {
        cudaError_t error = cudaMemcpy(
            (void *)this->d_keys, h_keys, arrayLength * sizeof(*h_keys), cudaMemcpyHostToDevice
        );
        checkCudaError(error);

        memoryCopiedToDevice = true;
    }

    /*
    Copies data from device to host.
    */
    void memoryCopyDeviceToHost(data_t *h_keys, uint_t arrayLength)
    {
        cudaError_t error = cudaMemcpy(
            h_keys, (void *)this->d_keys, arrayLength * sizeof(*h_keys), cudaMemcpyDeviceToHost
        );
        checkCudaError(error);
    }

    /*
    Provides a wrapper for private sort.
    */
    void sort(data_t *h_keys, uint_t arrayLength, order_t sortOrder)
    {
        if (arrayLength > this->arrayLength)
        {
            memoryDestroy();
            memoryAllocate(arrayLength);
        }
        if (!memoryCopiedToDevice)
        {
            memoryCopyHostToDevice(h_keys, arrayLength);
        }

        this->h_keys = h_keys;
        this->arrayLength = arrayLength;
        this->sortOrder = sortOrder;

        sortPrivate();
        memoryCopiedToDevice = false;

        if (memoryCopyAfterSort)
        {
            memoryCopyDeviceToHost(h_keys, arrayLength);
        }
    }
};


/*
Base class for parallel sort of key-value pairs.
*/
class SortParallelKeyValue : public SortParallelKeyOnly
{
protected:
    // Array for values on host
    data_t *h_values = NULL;
    // Array for values on device
    data_t *d_values = NULL;
    // Parallel sort for key-value pairs
    sort_type_t sortType = SORT_PARALLEL_KEY_VALUE;

public:
    /*
    Method for allocating memory needed for sort.
    */
    void memoryAllocate(uint_t arrayLength)
    {
        // Allocates keys
        SortParallelKeyOnly::memoryAllocate(arrayLength);
        // Allocates values
        cudaError_t error = cudaMalloc((void **)this->d_values, arrayLength * sizeof(*(this->d_values)));
        checkCudaError(error);
    }

    /*
    Method for destroying memory needed for sort.
    */
    void memoryDestroy()
    {
        // Destroys keys
        SortParallelKeyOnly::memoryDestroy();
        // Destroys values
        cudaError_t error = cudaFree(this->d_values);
        checkCudaError(error);
    }

    /*
    Copies data from host to device
    */
    void memoryCopyHostToDevice(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        // Copies keys
        SortParallelKeyOnly::memoryCopyHostToDevice(h_keys, arrayLength);
        // Copies values
        cudaError_t error = cudaMemcpy(
            (void *)this->d_keys, h_keys, arrayLength * sizeof(*h_keys), cudaMemcpyHostToDevice
        );
        checkCudaError(error);

        memoryCopiedToDevice = true;
    }

    /*
    Copies data from device to host.
    */
    void memoryCopyDeviceToHost(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        // Copies keys
        SortParallelKeyOnly::memoryCopyDeviceToHost(h_keys, arrayLength);
        // Copies values
        cudaError_t error = cudaMemcpy(
            h_values, (void *)this->d_values, arrayLength * sizeof(*h_values), cudaMemcpyDeviceToHost
        );
        checkCudaError(error);
    }

    /*
    Provides a wrapper for private sort.
    */
    void sort(data_t *h_keys, data_t *h_values, uint_t arrayLength, order_t sortOrder)
    {
        if (arrayLength > this->arrayLength)
        {
            memoryDestroy();
            memoryAllocate(arrayLength);
        }
        if (!memoryCopiedToDevice)
        {
            memoryCopyHostToDevice(h_keys, h_values, arrayLength);
        }

        this->h_keys = h_keys;
        this->h_values = h_values;
        this->arrayLength = arrayLength;
        this->sortOrder = sortOrder;

        sortPrivate();
        memoryCopiedToDevice = false;

        if (memoryCopyAfterSort)
        {
            memoryCopyDeviceToHost(h_keys, h_values, arrayLength);
        }
    }
};

#endif
