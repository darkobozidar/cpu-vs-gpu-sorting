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

    printf("|%d\n", node->value);

    level++;
    printBitonicTree(node->left, level);
    printBitonicTree(node->right, level);
}

void printBitonicTree(node_t *node)
{
    printBitonicTree(node, 0);
}

void bitonicTreeToArray(data_t *output, node_t *node, uint_t stride)
{
    output[0] = node->key;

    if (stride == 0)
    {
        return;
    }

    bitonicTreeToArray(output - stride, node->left, stride / 2);
    bitonicTreeToArray(output + stride, node->right, stride / 2);
}

void bitonicTreeToArray(data_t *output, node_t *root, node_t *spare, uint_t tableLen)
{
    if (tableLen == 1)
    {
        output[0] = root->key;
        return;
    }

    bitonicTreeToArray(output + tableLen / 2 - 1, root, tableLen / 4);
    output[tableLen - 1] = spare->key;
}

void swapNodeKeyValue(node_t *node1, node_t *node2)
{
    data_t temp;

    temp = node1->key;
    node1->key = node2->key;
    node2->key = temp;

    temp = node1->value;
    node1->value = node2->value;
    node2->value = temp;
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
    bool rightExchange = sortOrder == ORDER_ASC ? (root->key > spare->key) : (root->key < spare->key);
    if (!rightExchange)
    {
        rightExchange = root->key == spare->key && (
            sortOrder == ORDER_ASC ? root->value > spare->value : root->value < spare->value
        );
    }

    if (rightExchange)
    {
        swapNodeKeyValue(root, spare);
    }

    node_t *leftNode = root->left;
    node_t *rightNode = root->right;

    while (leftNode != NULL)
    {
        bool elementExchange = sortOrder == ORDER_ASC ? (leftNode->key > rightNode->key) : (leftNode->key < rightNode->key);
        if (!elementExchange)
        {
            elementExchange = leftNode->key == rightNode->key && (
                sortOrder == ORDER_ASC ? leftNode->value > rightNode->value : leftNode->value < rightNode->value
            );
        }

        if (rightExchange)
        {
            if (elementExchange)
            {
                swapNodeKeyValue(leftNode, rightNode);
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
                swapNodeKeyValue(leftNode, rightNode);
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

// TODO remove *dataTable pointer
void constructBitonicTree(data_t *dataTable, node_t *parent, uint_t stride, order_t sortOrder)
{
    if (stride == 0)
    {
        return;
    }

    node_t *leftNode = new node_t(dataTable[parent->value - stride], parent->value - stride);
    node_t *rightNode = new node_t(dataTable[parent->value + stride], parent->value + stride);

    parent->left = leftNode;
    parent->right = rightNode;

    constructBitonicTree(dataTable, leftNode, stride / 2, sortOrder);
    constructBitonicTree(dataTable, rightNode, stride / 2, (order_t)!sortOrder);
}

void adaptiveBitonicSort(node_t *root, node_t *spare, order_t sortOrder)
{
    if (root->left == NULL)
    {
        if (sortOrder == ORDER_ASC ? (root->key > spare->key) : (root->key < spare->key))
        {
            swapNodeKeyValue(root, spare);
        }
    }
    else
    {
        adaptiveBitonicSort(root->left, root, sortOrder);
        adaptiveBitonicSort(root->right, spare, (order_t)!sortOrder);
        bitonicMerge(root, spare, sortOrder);
    }
}

/*
Sorts data sequentially with adaptive bitonic sort.
*/
double sortSequential(data_t* output, data_t* buffer, uint_t tableLen, order_t sortOrder)
{
    LARGE_INTEGER timer;
    startStopwatch(&timer);

    node_t *root = new node_t(buffer[tableLen / 2 - 1], tableLen / 2 - 1);
    node_t *spare = new node_t(buffer[tableLen - 1], tableLen - 1);

    constructBitonicTree(buffer, root, tableLen / 4, sortOrder);
    adaptiveBitonicSort(root, spare, sortOrder);
    bitonicTreeToArray(output, root, spare, tableLen);

    double time = endStopwatch(timer);
    return time;
}
