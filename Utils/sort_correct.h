#ifndef SORT_CORRECT_H
#define SORT_CORRECT_H

#include "data_types_common.h"


template <typename T>
void quickSort(T *dataTable, uint_t tableLen, order_t sortOrder);

template <typename T>
void stdVectorSort(T *dataTable, uint_t tableLen, order_t sortOrder);

double sortCorrect(data_t *dataTable, uint_t tableLen, order_t sortOrder);

#endif
