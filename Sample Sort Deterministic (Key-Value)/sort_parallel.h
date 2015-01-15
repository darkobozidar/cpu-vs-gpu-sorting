#ifndef SORT_PARALLEL_H
#define SORT_PARALLEL_H


double sortParallel(
    data_t *h_keys, data_t *h_values, data_t *d_dataKeys, data_t *d_dataValues, data_t *d_bufferKeys,
    data_t *d_bufferValues, data_t *d_samplesLocal, data_t *d_samplesGlobal, uint_t *d_localBucketSizes,
    uint_t *d_localBucketOffsets, uint_t *h_globalBucketOffsets, uint_t *d_globalBucketOffsets,
    uint_t tableLen, order_t sortOrder
);

#endif
