#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <stdint.h>


// TODO comment
typedef struct Interval interval_t;
typedef struct Node node_t;

struct Interval
{
    uint32_t offset0;
    uint32_t length0;
    uint32_t offset1;
    uint32_t length1;
};

struct Node
{
    data_t *value;
    node_t *left;
    node_t *right;

    Node()
    {
        this->value = NULL;
        left = NULL;
        right = NULL;
    }

    Node(data_t *value)
    {
        this->value = value;
        left = NULL;
        right = NULL;
    }

    Node(data_t *value, node_t *left, node_t *right)
    {
        this->value = value;
        this->left = left;
        this->right = right;
    }
};

#endif
