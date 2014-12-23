#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "data_types.h"


/*
For debugging purposes prints out bitonic tree. Not to be called directly - bottom function calls it.
*/
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

/*
For debugging purposes prints out bitonic tree.
*/
void printBitonicTree(node_t *root)
{
    printBitonicTree(root, 0);
}

/*
Converts bitonic tree to array. Doesn't put value of spare node into array.
*/
void bitonicTreeToArray(data_t *output, node_t *node, uint_t stride)
{
    if (node->value >= 0)
    {
        output[0] = node->key;
    }

    if (stride == 0 || node->isDummyNode())
    {
        return;
    }

    bitonicTreeToArray(output - stride, node->left, stride / 2);
    bitonicTreeToArray(output + stride, node->right, stride / 2);
}

/*
Converts bitonic tree to array and puts value of spare node into array.
*/
void bitonicTreeToArray(data_t *output, node_t *root, node_t *spare, uint_t tableLen)
{
    if (tableLen == 1)
    {
        output[0] = root->key;
        return;
    }

    uint_t tableLenPower2 = nextPowerOf2(tableLen);
    uint_t padding = tableLenPower2 - tableLen;

    bitonicTreeToArray(output + tableLenPower2 / 2 - 1 - padding, root, tableLenPower2 / 4);
    output[tableLen - 1] = spare->key;
}

/*
Swaps node's key and value properties.
*/
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

/*
Swaps left nodes.
*/
void swapLeftNode(node_t *node1, node_t *node2)
{
    node_t *node = node1->left;
    node1->left = node2->left;
    node2->left = node;
}

/*
Swaps right nodes.
*/
void swapRightNode(node_t *node1, node_t *node2)
{
    node_t *node = node1->right;
    node1->right = node2->right;
    node2->right = node;
}

/*
Executes adaptive bitonic merge.

Adaptive bitonic merge works only for dictinct sequences. In case of duplicates in sequence values are compared
by their position in original (not sorted) array.
*/
void bitonicMerge(node_t *root, node_t *spare, order_t sortOrder)
{
    // Compares keys according to sort order
    bool rightExchange = sortOrder == ORDER_ASC ? (root->key > spare->key) : (root->key < spare->key);

    // In case of duplicates, ties are resolved according to element position in original unsorted array
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
        // Compares keys according to sort order
        bool elementExchange = sortOrder == ORDER_ASC ? (leftNode->key > rightNode->key) : (leftNode->key < rightNode->key);

        // In case of duplicates, ties are resolved according to element position in original unsorted array
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
        // TODO fix
        /*if (!root->left->isDummyNode())
        {
            bitonicMerge(root->left, root, sortOrder);
        }

        if (!root->right->isDummyNode())
        {
            bitonicMerge(root->right, spare, sortOrder);
        }*/

        bitonicMerge(root->left, root, sortOrder);
        bitonicMerge(root->right, spare, sortOrder);
    }
}

/*
Constructs bitonic tree from provided array.
Requires root node and stride (at beggining this is "<array_length> / 4)".
*/
template <data_t dummyValue>
void constructBitonicTree(data_t *dataTable, node_t *parent, int_t stride)
{
    // TODO fix (currently the whole padded tree is being built, NOT pruned tree)
    // if (stride == 0 || parent->value + 2 * stride <= 0)
    if (stride == 0)
    {
        return;
    }

    int_t newIndex = parent->value - stride;
    node_t *leftNode = new node_t(newIndex >= 0 ? dataTable[newIndex] : dummyValue, newIndex);
    newIndex = parent->value + stride;
    node_t *rightNode = new node_t(newIndex >= 0 ? dataTable[newIndex] : dummyValue, newIndex);

    parent->left = leftNode;
    parent->right = rightNode;

    constructBitonicTree<dummyValue>(dataTable, leftNode, stride / 2);
    constructBitonicTree<dummyValue>(dataTable, rightNode, stride / 2);
}

/*
Executes adaptive bitonic sort on provided bitonic tree. Requires root node of bitonic tree, spare node
(at beggining this is node with last array element with no children and parents) and sort order.
*/
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
        // TODO fix
        /*
        // If node does not represent a "dummy subtree" (doesn't contain dummy key and doesn't have both
        // pointers equal to NULL), then function is executed.

        if (!root->left->isDummyNode())
        {
            adaptiveBitonicSort(root->left, root, sortOrder);
        }

        if (!root->right->isDummyNode())
        {
            adaptiveBitonicSort(root->right, spare, (order_t)!sortOrder);
        }

        if (!root->isDummyNode())
        {
            bitonicMerge(root, spare, sortOrder);
        }*/

        adaptiveBitonicSort(root->left, root, sortOrder);
        adaptiveBitonicSort(root->right, spare, (order_t)!sortOrder);
        bitonicMerge(root, spare, sortOrder);
    }
}

/*
Sorts data sequentially with adaptive bitonic sort.
*/
double sortSequential(data_t* dataTable, uint_t tableLen, order_t sortOrder)
{
    LARGE_INTEGER timer;
    startStopwatch(&timer);

    uint_t tableLenPower2 = nextPowerOf2(tableLen);
    uint_t padding = tableLenPower2 - tableLen;

    node_t *root = new node_t(dataTable[tableLenPower2 / 2 - 1 - padding], tableLenPower2 / 2 - 1 - padding);
    node_t *spare = new node_t(dataTable[tableLen - 1], tableLen - 1);

    if (sortOrder == ORDER_ASC)
    {
        constructBitonicTree<MIN_VAL>(dataTable, root, tableLenPower2 / 4);
    }
    else
    {
        constructBitonicTree<MAX_VAL>(dataTable, root, tableLenPower2 / 4);
    }

    adaptiveBitonicSort(root, spare, sortOrder);
    bitonicTreeToArray(dataTable, root, spare, tableLen);

    double time = endStopwatch(timer);
    return time;
}
