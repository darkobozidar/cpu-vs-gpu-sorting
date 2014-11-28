#ifndef GENERATOR_H
#define GENERATOR_H

#include "data_types_common.h"


void fillTableKeysOnly(data_t *keys, uint_t tableLen, uint_t interval, data_dist_t distribution);
void fillTableKeysOnly(data_t *keys, uint_t tableLen, uint_t interval, uint_t bucketSize, data_dist_t distribution);
void fillTableKeyValue(data_t *keys, data_t *values, uint_t tableLen, uint_t interval, data_dist_t distribution);

#endif
