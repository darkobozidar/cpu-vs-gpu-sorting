#ifndef SORT_SEQUENTIAL_H
#define SORT_SEQUENTIAL_H


double sortSequential(
    data_t *inputKeys, data_t *inputValues, data_t *bufferKeys, data_t *bufferValues, data_t *outputKeys,
    data_t *outputValues, data_t *samples, uint_t *elementBuckets, uint_t tableLen, order_t sortOrder
);

#endif
