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

// TODO predelaj konstruktorje
struct Node
{
    data_t key;
    data_t value;
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
