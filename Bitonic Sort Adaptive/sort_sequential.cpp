#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "data_types.h"


void constructBitonicTree(node_t *parent, uint_t stride)
{
    if (stride == 0)
    {
        return;
    }

    node_t *leftNode = new node_t(parent->value - stride);
    node_t *rightNode = new node_t(parent->value + stride);

    parent->left = leftNode;
    parent->right = rightNode;

    constructBitonicTree(parent->left, stride / 2);
    constructBitonicTree(parent->right, stride / 2);
}

node_t* constructBitonicTree(data_t *dataTable, uint_t tableLen)
{
    node_t *root = new node_t(dataTable + tableLen / 2 - 1);
    constructBitonicTree(root, tableLen / 4);
    return root;
}


void printBitonicTree(node_t *node, uint_t level)
{
    if (node == NULL)
    {
        return;
    }

    for (uint_t i = 0; i < level; i++)
    {
        printf("  ");
    }

    printf("|%d\n", *node->value);

    level++;
    printBitonicTree(node->left, level);
    printBitonicTree(node->right, level);
}

void printBitonicTree(node_t *node)
{
    printBitonicTree(node, 0);
}


/*
Sorts data sequentially with NORMALIZED bitonic sort.
*/
double sortSequential(data_t* dataTable, uint_t tableLen, order_t sortOrder)
{
    node_t *root = constructBitonicTree(dataTable, tableLen);
    printBitonicTree(root);

    return 10;
}
