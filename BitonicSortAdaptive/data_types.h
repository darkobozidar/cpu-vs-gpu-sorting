#ifndef DATA_TYPES_BITONIC_SORT_ADAPTIVE_H
#define DATA_TYPES_BITONIC_SORT_ADAPTIVE_H

#include <stdint.h>


typedef struct Interval interval_t;
typedef struct Node node_t;

/*
Holds 2 intervals needed for IBR bitonic sort.
Intervals are represented with offset (index) in array and with length.
*/
struct Interval
{
    uint32_t offset0;
    uint32_t length0;
    uint32_t offset1;
    uint32_t length1;
};

/*
Represents a Node in bitonic tree needed for adaptive bitonic sort.

Adaptive bitonic sort works only for distinct sequences. If sequence isn't distinct, ties can be broken by the
element's original position in array. This is why this structure contains property "value" alongside property "key".
*/
struct Node
{
    data_t key;    // Holds value from array
    data_t value;   // Holds an index of element in original (not sorted) array
    node_t *left;
    node_t *right;

    Node(data_t key, data_t value, node_t *left, node_t *right)
    {

        this->key = key;
        this->value = value;
        this->left = left;
        this->right = right;
    }

    Node(data_t key, data_t value) : Node(key, value, NULL, NULL) {}

    Node(data_t key) : Node(key, key, NULL, NULL) {}

    Node() : Node(NULL, NULL, NULL, NULL) {}
};

#endif
