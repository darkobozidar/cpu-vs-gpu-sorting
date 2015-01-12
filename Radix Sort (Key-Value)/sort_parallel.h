#ifndef SORT_PARALLEL_H
#define SORT_PARALLEL_H


double sortParallel(
    data_t *h_keys, data_t *h_values, data_t *d_dataKeys, data_t *d_dataValues, data_t *d_bufferKeys,
    data_t *d_bufferValues, uint_t *d_bucketOffsetsLocal, uint_t *d_bucketOffsetsGlobal, uint_t *d_bucketSizes,
    uint_t tableLen, order_t sortOrder
);

#endif
