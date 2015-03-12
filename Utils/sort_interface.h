#ifndef SORT_INTERFACE_H
#define SORT_INTERFACE_H

#include <stdlib.h>
#include <stdio.h>
#include <string>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_types_common.h"
#include "host.h"
#include "cuda.h"


/*
Base class for sorting.
*/
class Sort
{
protected:
    // Length of array
    uint_t _arrayLength = 0;
    // Sort order (ascending or descending)
    order_t _sortOrder = ORDER_ASC;
    // Sort type
    sort_type_t _sortType = (sort_type_t)NULL;
    // Name of the sorting algorithm
    std::string _sortName = "Sort Name";
    // Time that sort needed for execution.
    double _sortTime = -1;
    // Denotes if sort timing should be executed.
    bool _stopwatchEnabled = false;

    /*
    Executes the sort.
    */
    virtual void sortPrivate() {}

public:
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

    void stopwatchEnable()
    {
        _stopwatchEnabled = true;
    }

    void stopwatchDisable()
    {
        _stopwatchEnabled = false;
    }

    double getSortTime()
    {
        if (!_stopwatchEnabled)
        {
            printf(
                "Stopwatch has to be explicitly enabled with method: stopwatchEnable().\n"
                "EXPLANATION: When sort is being timed, memory management isn't timed. In some sorting algorithms "
                "memory transfer is performed asynchroniously. When sort is timed, we have to wait for memory "
                "transfer to complete. After that sort can be performed. This takes MORE TIME than if the data "
                "was copied asynchroniously, that's why stopwatch has to be explicitly enabled.\n"
            );
            exit(EXIT_FAILURE);
        }

        if (_sortTime == -1)
        {
            printf("Sort hasn't been performed yet.\n");
            exit(EXIT_FAILURE);
        }

        return _sortTime;
    }

    /*
    Method for destroying memory needed for sort. For sort testing purposes this method is public.
    */
    virtual void memoryDestroy() {}

    /*
    Wrapper method, which executes all needed memory management and timing. Also calls private sort.
    Declared here for testing purposes, so all sorting algorithms can be placed in the same array.
    */
    virtual void sort(data_t *h_keys, uint_t arrayLength, order_t sortOrder) {}
    virtual void sort(data_t *h_keys, data_t *h_values, uint_t arrayLength, order_t sortOrder) {}
};


/*
Base class for sorts, which are sorting keys only.
*/
class SortKeyOnly : public Sort
{
protected:
    // Array of keys on host
    data_t *_h_keys = NULL;

    /*
    Method for allocating memory needed for sort.
    */
    virtual void memoryAllocate(data_t *h_keys, uint_t arrayLength)
    {
        _h_keys = h_keys;
        _arrayLength = arrayLength;
    }

    /*
    Memory copy operations needed before sort.
    */
    virtual void memoryCopyBeforeSort(data_t *h_keys, uint_t arrayLength) {}

    /*
    Memory copy operations needed after sort.
    */
    virtual void memoryCopyAfterSort(data_t *h_keys, uint_t arrayLength) {}

public:
    /*
    Wrapper method, which executes all needed memory management and timing. Also calls private sort.
    */
    virtual void sort(data_t *h_keys, uint_t arrayLength, order_t sortOrder)
    {
        if (arrayLength > _arrayLength)
        {
            memoryAllocate(h_keys, arrayLength);
        }

        memoryCopyBeforeSort(h_keys, arrayLength);
        _sortOrder = sortOrder;

        LARGE_INTEGER timer;
        if (_stopwatchEnabled)
        {
            if (getSortType() == SORT_PARALLEL_KEY_ONLY || getSortType() == SORT_PARALLEL_KEY_VALUE)
            {
                cudaDeviceSynchronize();
            }

            startStopwatch(&timer);
        }

        sortPrivate();

        if (_stopwatchEnabled)
        {
            if (getSortType() == SORT_PARALLEL_KEY_ONLY || getSortType() == SORT_PARALLEL_KEY_VALUE)
            {
                cudaDeviceSynchronize();
            }

            _sortTime = endStopwatch(timer);
        }

        memoryCopyAfterSort(h_keys, arrayLength);
    }
};


/*
Base class for sorts, which are sorting key-value pairs.
*/
class SortKeyValue : public Sort
{
protected:
    // Array of keys on host
    data_t *_h_keys = NULL;
    // Array of values on host
    data_t *_h_values = NULL;

    /*
    Method for allocating memory needed for sort.
    */
    virtual void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        _h_keys = h_keys;
        _arrayLength = arrayLength;
        _h_values = h_values;
    }

    /*
    Memory copy operations needed before sort.
    */
    virtual void memoryCopyBeforeSort(data_t *h_keys, data_t *h_values, uint_t arrayLength) {}

    /*
    Memory copy operations needed after sort.
    */
    virtual void memoryCopyAfterSort(data_t *h_keys, data_t *h_values, uint_t arrayLength) {}

public:
    /*
    Wrapper method, which executes all needed memory management and timing. Also calls private sort.
    */
    virtual void sort(data_t *h_keys, data_t *h_values, uint_t arrayLength, order_t sortOrder)
    {
        if (arrayLength > _arrayLength)
        {
            memoryAllocate(h_keys, h_values, arrayLength);
        }

        memoryCopyBeforeSort(h_keys, h_values, arrayLength);
        _sortOrder = sortOrder;

        LARGE_INTEGER timer;
        if (_stopwatchEnabled)
        {
            if (getSortType() == SORT_PARALLEL_KEY_ONLY || getSortType() == SORT_PARALLEL_KEY_VALUE)
            {
                cudaDeviceSynchronize();
            }

            startStopwatch(&timer);
        }

        sortPrivate();

        if (_stopwatchEnabled)
        {
            if (getSortType() == SORT_PARALLEL_KEY_ONLY || getSortType() == SORT_PARALLEL_KEY_VALUE)
            {
                cudaDeviceSynchronize();
            }

            _sortTime = endStopwatch(timer);
        }

        memoryCopyAfterSort(h_keys, h_values, arrayLength);
    }
};


//////////////////////////////////

/*
Base class for sequential sort of keys only.
*/
class SortSequentialKeyOnly : public SortKeyOnly
{
protected:
    // Sequential sort for keys only
    sort_type_t _sortType = SORT_SEQUENTIAL_KEY_ONLY;

public:
    sort_type_t getSortType()
    {
        return this->_sortType;
    }
};


/*
Base class for sequential sort of key-value pairs.
*/
class SortSequentialKeyValue : public SortKeyValue
{
protected:
    // Sequential sort for key-value pairs
    sort_type_t _sortType = SORT_SEQUENTIAL_KEY_VALUE;

public:
    sort_type_t getSortType()
    {
        return _sortType;
    }
};


/*
Base class for parallel sort of keys only.
*/
class SortParallelKeyOnly : public SortKeyOnly
{
protected:
    // Array for keys on device
    data_t *_d_keys = NULL;
    // Parallel sort for keys only
    sort_type_t _sortType = SORT_PARALLEL_KEY_ONLY;

    /*
    Method for allocating memory needed for sort.
    */
    void memoryAllocate(data_t *h_keys, uint_t arrayLength)
    {
        SortKeyOnly::memoryAllocate(h_keys, arrayLength);
        cudaError_t error = cudaMalloc((void **)&_d_keys, arrayLength * sizeof(*_d_keys));
        checkCudaError(error);
    }

    /*
    Copies data from host to device.
    */
    void memoryCopyBeforeSort(data_t *h_keys, uint_t arrayLength)
    {
        SortKeyOnly::memoryCopyBeforeSort(h_keys, arrayLength);
        cudaError_t error = cudaMemcpy(
            (void *)_d_keys, h_keys, arrayLength * sizeof(*h_keys), cudaMemcpyHostToDevice
        );
        checkCudaError(error);
    }

    /*
    Copies data from device to host.
    */
    void memoryCopyAfterSort(data_t *h_keys, uint_t arrayLength)
    {
        SortKeyOnly::memoryCopyAfterSort(h_keys, arrayLength);
        cudaError_t error = cudaMemcpy(
            h_keys, (void *)_d_keys, _arrayLength * sizeof(*_h_keys), cudaMemcpyDeviceToHost
        );
        checkCudaError(error);
    }

public:
    sort_type_t getSortType()
    {
        return _sortType;
    }

    /*
    Method for destroying memory needed for sort. For sort testing purposes this method is public.
    */
    void memoryDestroy()
    {
        SortKeyOnly::memoryDestroy();
        cudaError_t error = cudaFree(_d_keys);
        checkCudaError(error);
    }
};


/*
Base class for parallel sort of key-value pairs.
*/
class SortParallelKeyValue : public SortKeyValue
{
protected:
    // Array for keys on device
    data_t *_d_keys = NULL;
    // Array for values on device
    data_t *_d_values = NULL;
    // Parallel sort for key-value pairs
    sort_type_t _sortType = SORT_PARALLEL_KEY_VALUE;

    /*
    Method for allocating memory needed for sort.
    */
    void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        cudaError_t error;
        SortKeyValue::memoryAllocate(h_keys, h_values, arrayLength);

        // Allocates keys and values
        error = cudaMalloc((void **)&_d_keys, arrayLength * sizeof(*_d_keys));
        error = cudaMalloc((void **)&_d_values, _arrayLength * sizeof(*_d_values));
        checkCudaError(error);
    }

    /*
    Copies data from host to device.
    */
    void memoryCopyBeforeSort(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        cudaError_t error;
        SortKeyValue::memoryCopyBeforeSort(h_keys, h_values, arrayLength);

        // Copies keys and values
        error = cudaMemcpy(
            (void *)_d_keys, h_keys, arrayLength * sizeof(*h_keys), cudaMemcpyHostToDevice
        );
        checkCudaError(error);
        error = cudaMemcpy(
            (void *)_d_values, h_values, arrayLength * sizeof(*_d_values), cudaMemcpyHostToDevice
        );
        checkCudaError(error);
    }

    /*
    Copies data from device to host.
    */
    void memoryCopyAfterSort(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        cudaError_t error;
        SortKeyValue::memoryCopyAfterSort(h_keys, h_values, arrayLength);

        // Copies keys and values
        error = cudaMemcpy(
            h_keys, (void *)_d_keys, _arrayLength * sizeof(*_h_keys), cudaMemcpyDeviceToHost
        );
        checkCudaError(error);
        error = cudaMemcpy(
            h_values, (void *)_d_values, arrayLength * sizeof(*h_values), cudaMemcpyDeviceToHost
        );
        checkCudaError(error);
    }

public:
    sort_type_t getSortType()
    {
        return _sortType;
    }

    /*
    Method for destroying memory needed for sort. For sort testing purposes this method is public.
    */
    void memoryDestroy()
    {
        cudaError_t error;
        SortKeyValue::memoryDestroy();

        // Destroys keys and values
        error = cudaFree(_d_keys);
        error = cudaFree(_d_values);
        checkCudaError(error);
    }
};

#endif
