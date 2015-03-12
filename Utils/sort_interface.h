#ifndef SORT_INTERFACE_H
#define SORT_INTERFACE_H

#include <stdlib.h>
#include <stdio.h>
#include <string>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types_common.h"
#include "cuda.h"


/*
Base class for sorting.

For testing purposes memory management methods were made public, because data has to be copied from host to device,
which shouldn't be timed with stopwatch. In practice entire sort can be done only with method call sort(), because
memory management is already implemented in method sort.
*/
class Sort
{
protected:
    // Array of keys on host
    data_t *_h_keys = NULL;
    // Array of values on host
    data_t *_h_values = NULL;
    // Length of array
    uint_t _arrayLength = 0;
    // Sort order (ascending or descending)
    order_t _sortOrder = ORDER_ASC;
    // Designates if data should be copied from device to host after the sort is completed (needed for parallel sort)
    bool _memoryCopyAfterSort = true;
    // Sort type
    sort_type_t _sortType = (sort_type_t)NULL;
    // Name of the sorting algorithm
    std::string _sortName = "Sort Name";

    /*
    Executes the sort.
    */
    virtual void sortPrivate() {}

public:
    Sort() {}
    ~Sort()
    {
        memoryDestroy();
    }

    virtual sort_type_t getSortType()
    {
        return _sortType;
    }

    virtual std::string getSortName()
    {
        return _sortName;
    }

    // Needed for testing purposes
    void setMemoryCopyAfterSort(bool memoryCopyAfterSort)
    {
        _memoryCopyAfterSort = memoryCopyAfterSort;
    }

    /*
    Method for allocating memory needed for sort.
    */
    virtual void memoryAllocate(data_t *h_keys, uint_t arrayLength)
    {
        _h_keys = h_keys;
        _arrayLength = arrayLength;
    }
    virtual void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        memoryAllocate(h_keys, arrayLength);
        _h_values = h_values;
    }

    /*
    Method for destroying memory needed for sort.
    */
    virtual void memoryDestroy() {}

    /*
    Copies data from host to device. Needed only for parallel sorts.
    */
    virtual void memoryCopyHostToDevice(data_t *h_keys, uint_t arrayLength) {}
    virtual void memoryCopyHostToDevice(data_t *h_keys, data_t *h_values, uint_t arrayLength) {}

    /*
    Copies data from device to host. Needed only for parallel sorts.
    */
    virtual void memoryCopyDeviceToHost(data_t *h_keys, uint_t arrayLength) {}
    virtual void memoryCopyDeviceToHost(data_t *h_keys, data_t *h_values, uint_t arrayLength) {}

    /*
    Provides a wrapper for private sort.
    */
    virtual void sort(data_t *h_keys, uint_t arrayLength, order_t sortOrder) {}
    virtual void sort(data_t *h_keys, data_t *h_values, uint_t arrayLength, order_t sortOrder) {}
};


/*
Base class for sequential sort of keys only.
*/
class SortSequentialKeyOnly : public Sort
{
protected:
    // Sequential sort for keys only
    sort_type_t _sortType = SORT_SEQUENTIAL_KEY_ONLY;

public:
    sort_type_t getSortType()
    {
        return this->_sortType;
    }

    /*
    Provides a wrapper for private sort.
    */
    void sort(data_t *h_keys, uint_t arrayLength, order_t sortOrder)
    {
        if (arrayLength > _arrayLength)
        {
            memoryDestroy();
            memoryAllocate(h_keys, arrayLength);
        }

        _sortOrder = sortOrder;
        sortPrivate();
    }
};


/*
Base class for sequential sort of key-value pairs.
*/
class SortSequentialKeyValue : public Sort
{
protected:
    // Sequential sort for key-value pairs
    sort_type_t _sortType = SORT_SEQUENTIAL_KEY_VALUE;

public:
    sort_type_t getSortType()
    {
        return _sortType;
    }

    /*
    Provides a wrapper for private sort.
    */
    void sort(data_t *h_keys, data_t *h_values, uint_t arrayLength, order_t sortOrder)
    {
        if (arrayLength > _arrayLength)
        {
            memoryDestroy();
            memoryAllocate(h_keys, h_values, arrayLength);
        }

        _sortOrder = sortOrder;
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
    data_t *_d_keys = NULL;
    // Designates if data has been copied to device
    bool _memoryCopiedToDevice = false;
    // Parallel sort for keys only
    sort_type_t _sortType = SORT_PARALLEL_KEY_ONLY;

public:
    sort_type_t getSortType()
    {
        return _sortType;
    }

    /*
    Method for allocating memory needed for sort.
    */
    void memoryAllocate(data_t *h_keys, uint_t arrayLength)
    {
        Sort::memoryAllocate(h_keys, arrayLength);
        cudaError_t error = cudaMalloc((void **)&_d_keys, arrayLength * sizeof(*_d_keys));
        checkCudaError(error);
    }

    /*
    Method for destroying memory needed for sort.
    */
    void memoryDestroy()
    {
        cudaError_t error = cudaFree(_d_keys);
        checkCudaError(error);
    }

    /*
    Copies data from host to device.
    */
    void memoryCopyHostToDevice(data_t *h_keys, uint_t arrayLength)
    {
        cudaError_t error = cudaMemcpy(
            (void *)_d_keys, h_keys, arrayLength * sizeof(*h_keys), cudaMemcpyHostToDevice
        );
        checkCudaError(error);

        _memoryCopiedToDevice = true;
    }

    /*
    Copies data from device to host.
    */
    void memoryCopyDeviceToHost(data_t *h_keys, uint_t arrayLength)
    {
        cudaError_t error = cudaMemcpy(
            h_keys, (void *)_d_keys, _arrayLength * sizeof(*_h_keys), cudaMemcpyDeviceToHost
        );
        checkCudaError(error);
    }

    /*
    Provides a wrapper for private sort.
    */
    void sort(data_t *h_keys, uint_t arrayLength, order_t sortOrder)
    {
        if (arrayLength > _arrayLength)
        {
            memoryDestroy();
            memoryAllocate(h_keys, arrayLength);
        }
        if (!_memoryCopiedToDevice)
        {
            memoryCopyHostToDevice(h_keys, arrayLength);
        }

        _sortOrder = sortOrder;
        sortPrivate();
        _memoryCopiedToDevice = false;

        if (_memoryCopyAfterSort)
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
    data_t *_h_values = NULL;
    // Array for values on device
    data_t *_d_values = NULL;
    // Parallel sort for key-value pairs
    sort_type_t _sortType = SORT_PARALLEL_KEY_VALUE;

public:
    sort_type_t getSortType()
    {
        return _sortType;
    }

    /*
    Method for allocating memory needed for sort.
    */
    void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        // Allocates keys
        SortParallelKeyOnly::memoryAllocate(h_keys, arrayLength);
        // Allocates values
        cudaError_t error = cudaMalloc((void **)&_d_values, _arrayLength * sizeof(*_d_values));
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
        cudaError_t error = cudaFree(_d_values);
        checkCudaError(error);
    }

    /*
    Copies data from host to device.
    */
    void memoryCopyHostToDevice(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        // Copies keys
        SortParallelKeyOnly::memoryCopyHostToDevice(h_keys, arrayLength);
        // Copies values
        cudaError_t error = cudaMemcpy(
            (void *)_d_values, h_values, arrayLength * sizeof(*_d_values), cudaMemcpyHostToDevice
        );
        checkCudaError(error);

        _memoryCopiedToDevice = true;
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
            h_values, (void *)_d_values, arrayLength * sizeof(*h_values), cudaMemcpyDeviceToHost
        );
        checkCudaError(error);
    }

    /*
    Provides a wrapper for private sort.
    */
    void sort(data_t *h_keys, data_t *h_values, uint_t arrayLength, order_t sortOrder)
    {
        if (arrayLength > _arrayLength)
        {
            memoryDestroy();
            memoryAllocate(h_keys, h_values, arrayLength);
        }
        if (!_memoryCopiedToDevice)
        {
            memoryCopyHostToDevice(h_keys, h_values, arrayLength);
        }

        _sortOrder = sortOrder;
        sortPrivate();
        _memoryCopiedToDevice = false;

        if (_memoryCopyAfterSort)
        {
            memoryCopyDeviceToHost(h_keys, h_values, arrayLength);
        }
    }
};

#endif
