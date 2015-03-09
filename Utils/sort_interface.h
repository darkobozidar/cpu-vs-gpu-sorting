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
Base class for sequential sort of keys only.

Additional memory needed for sort has to be allocated on the instance of a class, because methods sort()
and sortPrivate don't have any parameters - sort is encapsulated in class.
*/
class SortSequentialKeyOnly
{
protected:
    // Array of keys on host
    data_t *h_keys = NULL;
    // Length of array
    uint_t arrayLength = 0;
    // Sort order (ascending or descending)
    order_t sortOrder;

    /*
    Executes the sort.
    */
    void virtual sortPrivate();

public:
    SortSequentialKeyOnly(data_t *h_keys, uint_t arrayLength, order_t sortOrder)
    {
        this->h_keys = h_keys;
        this->arrayLength = arrayLength;
        this->sortOrder = sortOrder;
    }

    SortSequentialKeyOnly()
    {
        SortSequentialKeyOnly(NULL, 0, ORDER_ASC);
    }

    ~SortSequentialKeyOnly();

    /*
    Provides a wrapper for private sort.
    */
    void sort()
    {
        sortPrivate();
    }
};


/*
Base class for sequential sort of key-value pairs.
*/
class SortSequentialKeyValue : public SortSequentialKeyOnly
{
protected:
    // Array of values on host
    data_t *h_values = NULL;

public:
    SortSequentialKeyValue(
        data_t *h_keys, data_t *h_values, uint_t arrayLength, order_t sortOrder
    ) : SortSequentialKeyOnly(h_keys, arrayLength, sortOrder)
    {
        this->h_values = h_values;
    }

    SortSequentialKeyValue()
    {
        SortSequentialKeyValue(NULL, NULL, 0, ORDER_ASC);
    }
};


/*
Base class for parallel sort of keys only.
*/
class SortParallelKeyOnly : public SortSequentialKeyOnly
{
protected:
    // Array for keys on device
    data_t *d_keys = NULL;
    // Denotes if data should be copied from device to host after the sort is completed
    bool memoryCopyAfterSort = true;

public:
    SortParallelKeyOnly(
        data_t *h_keys, uint_t arrayLength, order_t sortOrder, bool memoryCopyAfterSort
    ) : SortSequentialKeyOnly(h_keys, arrayLength, sortOrder)
    {
        cudaError_t error;
        this->memoryCopyAfterSort = memoryCopyAfterSort;

        // Allocated additional memory needed for sort
        error = cudaMalloc((void **)this->d_keys, arrayLength * sizeof(*(this->d_keys)));
        checkCudaError(error);

        // Copies data from host to device
        error = cudaMemcpy(
            (void *)this->d_keys, h_keys, arrayLength * sizeof(*h_keys), cudaMemcpyHostToDevice
        );
        checkCudaError(error);
    }

    SortParallelKeyOnly(data_t *h_keys, uint_t arrayLength, order_t sortOrder)
    {
        SortParallelKeyOnly(h_keys, arrayLength, sortOrder, true);
    }

    SortParallelKeyOnly(data_t *h_keys, uint_t arrayLength)
    {
        SortParallelKeyOnly(h_keys, arrayLength, ORDER_ASC, true);
    }

    SortParallelKeyOnly()
    {
        SortParallelKeyOnly(NULL, 0, ORDER_ASC, true);
    }

    ~SortParallelKeyOnly()
    {
        cudaError_t error = cudaFree(this->d_keys);
        checkCudaError(error);
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
    void sort()
    {
        sortPrivate();

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

public:
    SortParallelKeyValue(
        data_t *h_keys, data_t *h_values, uint_t arrayLength, order_t sortOrder, bool memoryCopyAfterSort
    ) : SortParallelKeyOnly(h_keys, arrayLength, sortOrder, memoryCopyAfterSort)
    {
        cudaError_t error;
        this->h_values = h_values;

        // Allocated additional memory needed for sort
        error = cudaMalloc((void **)this->d_values, arrayLength * sizeof(*(this->d_values)));
        checkCudaError(error);

        // Copies data from host to device
        error = cudaMemcpy(
            (void *)this->d_values, h_values, arrayLength * sizeof(*d_values), cudaMemcpyHostToDevice
        );
        checkCudaError(error);
    }

    SortParallelKeyValue(
        data_t *h_keys, data_t *h_values, uint_t arrayLength, order_t sortOrder
    )
    {
        SortParallelKeyValue(h_keys, h_values, arrayLength, sortOrder, true);
    }

    SortParallelKeyValue(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        SortParallelKeyValue(h_keys, h_values, arrayLength, ORDER_ASC, true);
    }

    SortParallelKeyValue()
    {
        SortParallelKeyValue(NULL, NULL, 0, ORDER_ASC, true);
    }

    ~SortParallelKeyValue()
    {
        cudaError_t error = cudaFree(this->d_values);
        checkCudaError(error);
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
    void sort()
    {
        sortPrivate();

        if (memoryCopyAfterSort)
        {
            memoryCopyDeviceToHost(h_keys, h_values, arrayLength);
        }
    }
};

#endif
