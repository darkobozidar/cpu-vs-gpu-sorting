#ifndef BITONIC_SORT_ADAPTIVE_SEQUENTIAL_H
#define BITONIC_SORT_ADAPTIVE_SEQUENTIAL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"
#include "../Utils/host.h"
#include "data_types.h"


/*
Base class for sequential adaptive bitonic sort.
TODO: reimplement without padding. In previous Git commits it is partially reimplemented without padding.
*/
class BitonicSortAdaptiveSequential : public SortSequential
{
protected:
    std::string _sortName = "Bitonic sort adaptive sequential";

    /*
    For debugging purposes prints out bitonic tree. Not to be called directly - bottom method calls it.
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
    Converts bitonic tree to array of keys. Doesn't put value of spare node into array.
    Not to be called directly - bottom function calls it.
    */
    void bitonicTreeToArrayKeyOnly(data_t *output, node_t *node, uint_t stride)
    {
        output[0] = node->key;

        if (stride == 0 || node->isDummyNode())
        {
            return;
        }

        bitonicTreeToArrayKeyOnly(output - stride, node->left, stride / 2);
        bitonicTreeToArrayKeyOnly(output + stride, node->right, stride / 2);
    }

    /*
    Converts bitonic tree to array of keys and values. Doesn't put value of spare node into array.
    Not to be called directly - bottom function calls it.
    */
    void bitonicTreeToArrayKeyValue(data_t *keys, data_t *values, node_t *node, uint_t stride)
    {
        keys[0] = node->key;
        values[0] = node->value;

        if (stride == 0 || node->isDummyNode())
        {
            return;
        }

        bitonicTreeToArrayKeyValue(keys - stride, values - stride, node->left, stride / 2);
        bitonicTreeToArrayKeyValue(keys + stride, values + stride, node->right, stride / 2);
    }

    /*
    Converts bitonic tree to array and puts value of spare node into array.
    */
    template <bool sortingKeyOnly>
    void bitonicTreeToArray(data_t *keys, data_t *values, node_t *root, node_t *spare, uint_t tableLen)
    {
        if (tableLen == 1)
        {
            keys[0] = root->key;
            values[0] = root->value;
            return;
        }

        uint_t tableLenPower2 = nextPowerOf2(tableLen);
        uint_t padding = tableLenPower2 - tableLen;

        if (sortingKeyOnly)
        {
            bitonicTreeToArrayKeyOnly(keys + tableLenPower2 / 2 - 1 - padding, root, tableLenPower2 / 4);
        }
        else
        {
            bitonicTreeToArrayKeyValue(
                keys + tableLenPower2 / 2 - 1 - padding, values + tableLenPower2 / 2 - 1 - padding, root,
                tableLenPower2 / 4
            );
        }

        // Inserts spare node
        keys[tableLen - 1] = spare->key;
        if (!sortingKeyOnly)
        {
            values[tableLen - 1] = spare->value;
        }
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
    template <order_t sortOrder>
    void bitonicMerge(node_t *root, node_t *spare)
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
            bitonicMerge<sortOrder>(root->left, root);
            bitonicMerge<sortOrder>(root->right, spare);
        }
    }

    /*
    Constructs bitonic tree from provided array of keys.
    Requires root node and stride (at beggining this is "<array_length> / 4").
    */
    void constructBitonicTreeKeyOnly(data_t *keys, node_t *parent, int_t stride)
    {
        if (stride == 0)
        {
            return;
        }

        int_t newIndex = parent->value - stride;
        parent->left = new node_t(keys[newIndex], newIndex);
        newIndex = parent->value + stride;
        parent->right = new node_t(keys[newIndex], newIndex);

        constructBitonicTreeKeyOnly(keys, parent->left, stride / 2);
        constructBitonicTreeKeyOnly(keys, parent->right, stride / 2);
    }

    /*
    Constructs bitonic tree from provided array of keys and values.
    Requires root node and stride (at beggining this is "<array_length> / 4").
    */
    void constructBitonicTreeKeyValue(data_t *keys, data_t *values, node_t *parent, int_t stride)
    {
        if (stride == 0)
        {
            return;
        }

        int_t newIndex = parent->value - stride;
        parent->left = new node_t(keys[newIndex], values[newIndex]);
        newIndex = parent->value + stride;
        parent->right = new node_t(keys[newIndex], values[newIndex]);

        constructBitonicTreeKeyValue(keys, values, parent->left, stride / 2);
        constructBitonicTreeKeyValue(keys, values, parent->right, stride / 2);
    }

    /*
    Executes adaptive bitonic sort on provided bitonic tree. Requires root node of bitonic tree, spare node
    (at beggining this is node with last array element with no children and parents) and sort order.
    */
    template <order_t sortOrder>
    void bitonicSortAdaptiveSequential(node_t *root, node_t *spare)
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
            bitonicSortAdaptiveSequential<sortOrder>(root->left, root);
            bitonicSortAdaptiveSequential<(order_t)!sortOrder>(root->right, spare);
            bitonicMerge<sortOrder>(root, spare);
        }
    }

    /*
    Constructs bitonic tree and sorts data sequentially with adaptive bitonic sort.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void adaptiveBitonicSortWrapper(data_t* keys, data_t *values, uint_t tableLen)
    {
        uint_t tableLenPower2 = nextPowerOf2(tableLen);
        uint_t padding = tableLenPower2 - tableLen;
        uint_t rootIndex = tableLenPower2 / 2 - 1 - padding;

        node_t *root = new node_t(keys[rootIndex], sortingKeyOnly ? rootIndex : values[rootIndex]);
        node_t *spare = new node_t(keys[tableLen - 1], sortingKeyOnly ? tableLen - 1 : values[tableLen - 1]);

        if (sortingKeyOnly)
        {
            constructBitonicTreeKeyOnly(keys, root, tableLenPower2 / 4);
        }
        else
        {
            constructBitonicTreeKeyValue(keys, values, root, tableLenPower2 / 4);
        }

        bitonicSortAdaptiveSequential<sortOrder>(root, spare);
        bitonicTreeToArray<sortingKeyOnly>(keys, values, root, spare, tableLen);
    };

    /*
    Wrapper for bitonic sort method.
    The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    */
    void sortKeyOnly()
    {
        if (_sortOrder == ORDER_ASC)
        {
            adaptiveBitonicSortWrapper<ORDER_ASC, true>(_h_keys, NULL, _arrayLength);
        }
        else
        {
            adaptiveBitonicSortWrapper<ORDER_DESC, true>(_h_keys, NULL, _arrayLength);
        }
    }

    /*
    Wrapper for bitonic sort method.
    The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    */
    void sortKeyValue()
    {
        if (_sortOrder == ORDER_ASC)
        {
            adaptiveBitonicSortWrapper<ORDER_ASC, false>(_h_keys, _h_values, _arrayLength);
        }
        else
        {
            adaptiveBitonicSortWrapper<ORDER_DESC, false>(_h_keys, _h_values, _arrayLength);
        }
    }

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};

#endif
