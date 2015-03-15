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
    // Root node of bitonic tree
    node_t _root = NULL;
    // Spare node of bitonic tree (right most element of array)
    node_t _spare = NULL;

    /*
    Method for allocating memory needed both for key only and key-value sort.
    */
    virtual void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        cudaError_t error;
        SortSequential::memoryAllocate(h_keys, h_values, arrayLength);
    }

    /*
    Memory copy operations needed before sort. If sorting keys only, than "h_values" contains NULL.
    */
    virtual void memoryCopyBeforeSort(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        cudaError_t error;
        SortSequential::memoryCopyBeforeSort(h_keys, h_values, arrayLength);
    }

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

        printf("|%d\n", node->key);

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
    void bitonicTreeToArrayKeyOnly(data_t *h_keys, node_t *node, uint_t arrayLength, uint_t index, uint_t stride)
    {
        if (index < arrayLength)
        {
            h_keys[index] = node->key;
        }

        if (stride == 0)
        {
            return;
        }

        bitonicTreeToArrayKeyOnly(h_keys, node->left, arrayLength, index - stride, stride / 2);
        if (index < arrayLength)
        {
            bitonicTreeToArrayKeyOnly(h_keys, node->right, arrayLength, index + stride, stride / 2);
        }
    }

    /*
    Converts bitonic tree to array of keys and values. Doesn't put value of spare node into array.
    Not to be called directly - bottom function calls it.
    */
    void bitonicTreeToArrayKeyValue(
        data_t *keys, data_t *values, node_t *node, uint_t arrayLength, uint_t index, uint_t stride
    )
    {
        if (index < arrayLength)
        {
            keys[index] = node->key;
            values[index] = node->value;
        }

        if (stride == 0)
        {
            return;
        }

        bitonicTreeToArrayKeyValue(keys, values, node->left, arrayLength, index - stride, stride / 2);
        if (index < arrayLength)
        {
            bitonicTreeToArrayKeyValue(keys, values, node->right, arrayLength, index + stride, stride / 2);
        }
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

        if (sortingKeyOnly)
        {
            bitonicTreeToArrayKeyOnly(keys, root, tableLen, tableLenPower2 / 2 - 1, tableLenPower2 / 4);
        }
        else
        {
            bitonicTreeToArrayKeyValue(
                keys, values, root, tableLen, tableLenPower2 / 2 - 1, tableLenPower2 / 4
            );
        }

        if (tableLen < tableLenPower2)
        {
            return;
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
    template <data_t dummyValue>
    void constructBitonicTreeKeyOnly(data_t *keys, node_t *parent, uint_t arrayLength, int_t stride)
    {
        if (stride == 0)
        {
            return;
        }

        int_t newIndex = parent->value - stride;
        parent->left = new node_t(newIndex < arrayLength ? keys[newIndex] : dummyValue, newIndex);
        newIndex = parent->value + stride;
        parent->right = new node_t(newIndex < arrayLength ? keys[newIndex] : dummyValue, newIndex);

        constructBitonicTreeKeyOnly<dummyValue>(keys, parent->left, arrayLength, stride / 2);
        constructBitonicTreeKeyOnly<dummyValue>(keys, parent->right, arrayLength, stride / 2);
    }

    /*
    Constructs bitonic tree from provided array of keys and values.
    Requires root node and stride (at beggining this is "<array_length> / 4").
    */
    template <data_t dummyValue>
    void constructBitonicTreeKeyValue(data_t *keys, data_t *values, node_t *parent, uint_t arrayLength, int_t stride)
    {
        if (stride == 0)
        {
            return;
        }

        int_t newIndex = parent->value - stride;
        parent->left = new node_t(
            newIndex < arrayLength ? keys[newIndex] : dummyValue,
            newIndex < arrayLength ? values[newIndex] : newIndex
        );
        newIndex = parent->value + stride;
        parent->right = new node_t(
            newIndex < arrayLength ? keys[newIndex] : dummyValue,
            newIndex < arrayLength ? values[newIndex] : newIndex
        );

        constructBitonicTreeKeyValue<dummyValue>(keys, values, parent->left, arrayLength, stride / 2);
        constructBitonicTreeKeyValue<dummyValue>(keys, values, parent->right, arrayLength, stride / 2);
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
    Deletes bitonic tree.
    */
    void deleteBitonicTree(node_t *node)
    {
        if (node == NULL)
        {
            return;
        }

        deleteBitonicTree(node->left);
        deleteBitonicTree(node->right);
        delete node;
    }

    /*
    Constructs bitonic tree and sorts data sequentially with adaptive bitonic sort.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void adaptiveBitonicSortWrapper(data_t* keys, data_t *values, uint_t tableLen)
    {
        uint_t tableLenPower2 = nextPowerOf2(tableLen);
        uint_t rootIndex = tableLenPower2 / 2 - 1;

        node_t *root = new node_t(keys[rootIndex], sortingKeyOnly ? rootIndex : values[rootIndex]);
        node_t *spare;

        if (tableLen == tableLenPower2)
        {
            spare = new node_t(
                keys[tableLenPower2 - 1], sortingKeyOnly ? tableLen - 1 : values[tableLen - 1]
            );
        }
        else
        {
            spare = new node_t(sortOrder == ORDER_ASC ? MAX_VAL : MIN_VAL, tableLenPower2);
        }

        if (sortingKeyOnly)
        {
            constructBitonicTreeKeyOnly<sortOrder == ORDER_ASC ? MAX_VAL : MIN_VAL>(
                keys, root, tableLen, tableLenPower2 / 4
            );
        }
        else
        {
            constructBitonicTreeKeyValue<sortOrder == ORDER_ASC ? MAX_VAL : MIN_VAL>(
                keys, values, root, tableLen, tableLenPower2 / 4
            );
        }

        bitonicSortAdaptiveSequential<sortOrder>(root, spare);
        bitonicTreeToArray<sortingKeyOnly>(keys, values, root, spare, tableLen);

        deleteBitonicTree(root);
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
