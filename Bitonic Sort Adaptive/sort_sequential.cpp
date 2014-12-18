#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "data_types.h"


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

void bitonicTreeToArray(data_t *dataTable, node_t *node, uint_t stride)
{
    // TODO
}

void swapNodeValues(node_t *node1, node_t *node2)
{
    data_t value = *node1->value;
    *node1->value = *node2->value;
    *node2->value = value;
}

void swapLeftNode(node_t *node1, node_t *node2)
{
    node_t *node = node1->left;
    node1->left = node2->left;
    node2->left = node;
}

void swapRightNode(node_t *node1, node_t *node2)
{
    node_t *node = node1->right;
    node1->right = node2->right;
    node2->right = node;
}

void bitonicMerge(node_t *root, node_t *spare, order_t sortOrder)
{
    bool rightExchange = (*root->value > *spare->value) ^ sortOrder;

    if (rightExchange)
    {
        swapNodeValues(root, spare);
    }

    node_t *leftNode = root->left;
    node_t *rightNode = root->right;

    while (leftNode != NULL)
    {
        bool elementExchange = (*leftNode->value > *rightNode->value) ^ sortOrder;

        if (rightExchange)
        {
            if (elementExchange)
            {
                swapNodeValues(leftNode, rightNode);
                swapRightNode(leftNode, rightNode);

                leftNode = leftNode->left;
                rightNode = rightNode->left;
            }
            else
            {
                leftNode = leftNode->right;
                rightNode = rightNode->right;
            }
        }
        else
        {
            if (elementExchange)
            {
                swapNodeValues(leftNode, rightNode);
                swapLeftNode(leftNode, rightNode);

                leftNode = leftNode->right;
                rightNode = rightNode->right;
            }
            else
            {
                leftNode = leftNode->left;
                rightNode = rightNode->left;
            }
        }
    }

    if (root->left != NULL)
    {
        bitonicMerge(root->left, root, sortOrder);
        bitonicMerge(root->right, spare, sortOrder);
    }
}

void compareExchange(data_t *value1, data_t *value2, order_t sortOrder)
{
    if ((*value1 > *value2) ^ sortOrder)
    {
        data_t temp = *value1;
        *value1 = *value2;
        *value2 = temp;
    }
}

void bitonicSort(data_t *dataTable, node_t *parent, uint_t stride, order_t sortOrder)
{
    if (stride == 0)
    {
        compareExchange(parent->value, parent->value + 1, sortOrder);
        return;
    }

    node_t *leftNode = new node_t(parent->value - stride);
    node_t *rightNode = new node_t(parent->value + stride);

    parent->left = leftNode;
    parent->right = rightNode;

    bitonicSort(dataTable, parent->left, stride / 2, sortOrder);
    bitonicSort(dataTable, parent->right, stride / 2, (order_t)!sortOrder);

    bitonicMerge(parent, new node_t(parent->value + 2 * stride, NULL, NULL), sortOrder);
}

void bitonicSort(data_t *dataTable, uint_t tableLen, order_t sortOrder)
{
    node_t *root = new node_t(dataTable + tableLen / 2 - 1);
    bitonicSort(dataTable, root, tableLen / 4, sortOrder);
    printBitonicTree(root);
}

/*
Sorts data sequentially with NORMALIZED bitonic sort.
*/
double sortSequential(data_t* dataTable, uint_t tableLen, order_t sortOrder)
{
    bitonicSort(dataTable, tableLen, sortOrder);
    return 10;
}
