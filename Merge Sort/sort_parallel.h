#ifndef SORT_PARALLEL_H
#define SORT_PARALLEL_H

void sortParallel(data_t *h_inputKeys, data_t *h_inputVals, data_t *h_outputKeys, data_t *h_outputVals,
                  uint_t arrayLen, bool orderAsc);

#endif
