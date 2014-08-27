#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <stdint.h>

typedef int data_t;
typedef uint32_t uint_t;

// TODO comment
struct SampleElement {
	data_t sample;
	uint_t rank;
};
typedef struct SampleElement sample_el_t;

#endif
