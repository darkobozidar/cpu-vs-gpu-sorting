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
Base class for sorts, which are sorting keys only.
*/
class SortSequential
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
    // Name of the sorting algorithm
    std::string _sortName = "Sort Name";
    // Denotes if sort is sequential or parallel
    bool _isSortParallel = false;
    // Time that sort needed for execution.
    double _sortTime = -1;
    // Denotes if sort timing should be executed
    bool _stopwatchEnabled = false;

    /*
    Executes the sort.
    */
    virtual void sortKeyOnly()
    {
        printf("Method sortKeyOnly() not implemented\n.");
        exit(EXIT_FAILURE);
    }
    virtual void sortKeyValue()
    {
        printf("Method sortKeyValue() not implemented\n.");
        exit(EXIT_FAILURE);
    }

    /*
    Sets private variables when sort() is called.
    */
    virtual void setPrivateVars(data_t *h_keys, data_t *h_values, uint_t arrayLength, order_t sortOrder)
    {
        _h_keys = h_keys;
        _h_values = h_values;
        _arrayLength = arrayLength;
        _sortOrder = sortOrder;
    }

    /*
    Method for allocating memory needed both for key only and key-value sort.
    */
    virtual void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength) {}

    /*
    Memory copy operations needed before sort. If sorting keys only, than "h_values" contains NULL.
    */
    virtual void memoryCopyBeforeSort(data_t *h_keys, data_t *h_values, uint_t arrayLength) {}

    /*
    Memory copy operations needed after sort. If sorting keys only, than "h_values" contains NULL.
    */
    virtual void memoryCopyAfterSort(data_t *h_keys, data_t *h_values, uint_t arrayLength) {}

public:
    ~SortSequential()
    {
        memoryDestroy();
    }

    virtual std::string getSortName()
    {
        return _sortName;
    }

    virtual std::string getSortName(bool sortingKeyOnly)
    {
        if (sortingKeyOnly)
        {
            return getSortName() + " key only";
        }

        return getSortName() + " key value";
    }

    virtual bool isSortParallel()
    {
        return _isSortParallel;
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
    virtual void memoryDestroy()
    {
        _arrayLength = 0;
    }

    /*
    Wrapper method, which executes all needed memory management and timing. Also calls private sort.
    */
    virtual void sort(data_t *h_keys, uint_t arrayLength, order_t sortOrder)
    {
        cudaError_t error;

        if (arrayLength > _arrayLength)
        {
            memoryAllocate(h_keys, NULL, arrayLength);
        }

        setPrivateVars(h_keys, NULL, arrayLength, sortOrder);
        memoryCopyBeforeSort(h_keys, NULL, arrayLength);

        LARGE_INTEGER timer;
        if (_stopwatchEnabled)
        {
            if (isSortParallel())
            {
                error = cudaDeviceSynchronize();
                checkCudaError(error);
            }

            startStopwatch(&timer);
        }

        sortKeyOnly();

        if (_stopwatchEnabled)
        {
            if (isSortParallel())
            {
                error = cudaDeviceSynchronize();
                checkCudaError(error);
            }

            _sortTime = endStopwatch(timer);
        }

        memoryCopyAfterSort(h_keys, NULL, arrayLength);
    }

    /*
    Wrapper method, which executes all needed memory management and timing. Also calls private sort.
    */
    virtual void sort(data_t *h_keys, data_t *h_values, uint_t arrayLength, order_t sortOrder)
    {
        cudaError_t error;

        if (arrayLength > _arrayLength)
        {
            memoryAllocate(h_keys, h_values, arrayLength);
        }

        setPrivateVars(h_keys, h_values, arrayLength, sortOrder);
        memoryCopyBeforeSort(h_keys, h_values, arrayLength);

        LARGE_INTEGER timer;
        if (_stopwatchEnabled)
        {
            if (isSortParallel())
            {
                error = cudaDeviceSynchronize();
                checkCudaError(error);
            }

            startStopwatch(&timer);
        }

        sortKeyValue();

        if (_stopwatchEnabled)
        {
            if (isSortParallel())
            {
                error = cudaDeviceSynchronize();
                checkCudaError(error);
            }

            _sortTime = endStopwatch(timer);
        }

        memoryCopyAfterSort(h_keys, h_values, arrayLength);
    }
};


/*
Base class for parallel sort of key-value pairs.
*/
class SortParallel : public SortSequential
{
protected:
    // Array for keys on device
    data_t *_d_keys = NULL;
    // Array for values on device
    data_t *_d_values = NULL;
    // Denotes if sort is sequential or parallel
    bool _isSortParallel = true;

    /*
    Method for allocating memory needed both for key only and key-value sort.
    */
    virtual void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        cudaError_t error;
        SortSequential::memoryAllocate(h_keys, h_values, arrayLength);

        // Allocates keys and values
        error = cudaMalloc((void **)&_d_keys, arrayLength * sizeof(*_d_keys));
        checkCudaError(error);
        error = cudaMalloc((void **)&_d_values, arrayLength * sizeof(*_d_values));
        checkCudaError(error);
    }

    /*
    Memory copy operations needed before sort. If sorting keys only, than "h_values" contains NULL.
    */
    virtual void memoryCopyBeforeSort(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        cudaError_t error;
        SortSequential::memoryCopyBeforeSort(h_keys, h_values, arrayLength);

        // Copies keys
        error = cudaMemcpy(
            (void *)_d_keys, h_keys, arrayLength * sizeof(*h_keys), cudaMemcpyHostToDevice
        );
        checkCudaError(error);

        if (h_values == NULL)
        {
            return;
        }

        // Copies values
        error = cudaMemcpy(
            (void *)_d_values, h_values, arrayLength * sizeof(*_d_values), cudaMemcpyHostToDevice
        );
        checkCudaError(error);
    }

    /*
    Copies data from device to host. If sorting keys only, than "h_values" contains NULL.
    */
    virtual void memoryCopyAfterSort(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        cudaError_t error;
        SortSequential::memoryCopyAfterSort(h_keys, h_values, arrayLength);

        // Copies keys
        error = cudaMemcpy(
            h_keys, (void *)_d_keys, _arrayLength * sizeof(*_h_keys), cudaMemcpyDeviceToHost
        );
        checkCudaError(error);

        if (h_values == NULL)
        {
            return;
        }

        // Copies values
        error = cudaMemcpy(
            h_values, (void *)_d_values, arrayLength * sizeof(*h_values), cudaMemcpyDeviceToHost
        );
        checkCudaError(error);
    }

public:
    bool isSortParallel()
    {
        return _isSortParallel;
    }

    /*
    Method for destroying memory needed for sort. For sort testing purposes this method is public.
    */
    virtual void memoryDestroy()
    {
        if (_arrayLength == 0)
        {
            return;
        }

        cudaError_t error;
        SortSequential::memoryDestroy();

        // Destroys keys and values
        error = cudaFree(_d_keys);
        checkCudaError(error);
        error = cudaFree(_d_values);
        checkCudaError(error);
    }
};

#endif
