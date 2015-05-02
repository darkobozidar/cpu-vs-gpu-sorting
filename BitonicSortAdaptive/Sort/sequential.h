#ifndef BITONIC_SORT_ADAPTIVE_SEQUENTIAL_H
#define BITONIC_SORT_ADAPTIVE_SEQUENTIAL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../../Utils/data_types_common.h"
#include "../../Utils/sort_interface.h"
#include "../../Utils/host.h"
#include "../data_types.h"


/*
Class for sequential adaptive bitonic sort.
TODO: reimplement without padding. In previous Git commits it is partially reimplemented without padding.
*/
class BitonicSortAdaptiveSequential : public SortSequential
{
protected:
    std::string _sortName = "Bitonic sort adaptive sequential";
    // Root node of bitonic tree
    node_t *_root = NULL;
    // Spare node of bitonic tree (right most element of array)
    node_t *_spare = NULL;

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
    Requires root node and stride (at beginning this is "<array_length> / 4").
    */
    void constructBitonicTree(node_t *parent, int_t stride)
    {
        if (stride == 0)
        {
            return;
        }

        parent->left = new node_t();
        parent->right = new node_t();

        constructBitonicTree(parent->left, stride / 2);
        constructBitonicTree(parent->right, stride / 2);
    }

    /*
    Method for allocating memory needed both for key only and key-value sort.
    */
    virtual void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        SortSequential::memoryAllocate(h_keys, h_values, arrayLength);

        _root = new Node();
        _spare = new Node();

        constructBitonicTree(_root, nextPowerOf2(arrayLength) / 4);
    }

    /*
    Fills bitonic tree with keys from array and generates unique values.
    */
    template <data_t minMaxValue>
    void fillBitonicTreeKeyOnly(data_t *keys, node_t *node, uint_t arrayLength, data_t arrayIndex, int_t stride)
    {
        if (node == NULL)
        {
            return;
        }

        node->key = arrayIndex < arrayLength ? keys[arrayIndex] : minMaxValue;
        node->value = arrayIndex;

        fillBitonicTreeKeyOnly<minMaxValue>(keys, node->left, arrayLength, arrayIndex - stride, stride / 2);
        fillBitonicTreeKeyOnly<minMaxValue>(keys, node->right, arrayLength, arrayIndex + stride, stride / 2);
    }

    /*
    Fills bitonic tree with keys and values.
    */
    template <data_t minMaxValue>
    void fillBitonicTreeKeyValue(
        data_t *h_keys, data_t *h_values, node_t *node, uint_t arrayLength, uint_t arrayIndex, int_t stride
    )
    {
        if (node == NULL)
        {
            return;
        }

        node->key = arrayIndex < arrayLength ? h_keys[arrayIndex] : minMaxValue;
        node->value = arrayIndex < arrayLength ? h_values[arrayIndex] : arrayIndex;

        fillBitonicTreeKeyValue<minMaxValue>(
            h_keys, h_values, node->left, arrayLength, arrayIndex - stride, stride / 2
        );
        fillBitonicTreeKeyValue<minMaxValue>(
            h_keys, h_values, node->right, arrayLength, arrayIndex + stride, stride / 2
        );
    }

    /*
    Memory copy operations needed before sort. If sorting keys only, than "h_values" contains NULL.
    */
    virtual void memoryCopyBeforeSort(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        SortSequential::memoryCopyBeforeSort(h_keys, h_values, arrayLength);

        uint_t arrayLenPowerOf2 = nextPowerOf2(arrayLength);
        uint_t rootIndex = arrayLenPowerOf2 / 2 - 1;
        bool sortingKeyOnly = h_values == NULL;

        if (sortingKeyOnly)
        {
            if (_sortOrder == ORDER_ASC)
            {
                fillBitonicTreeKeyOnly<MAX_VAL>(h_keys, _root, arrayLength, rootIndex, arrayLenPowerOf2 / 4);
            }
            else
            {
                fillBitonicTreeKeyOnly<MIN_VAL>(h_keys, _root, arrayLength, rootIndex, arrayLenPowerOf2 / 4);
            }
        }
        else
        {
            if (_sortOrder == ORDER_ASC)
            {
                fillBitonicTreeKeyValue<MAX_VAL>(
                    h_keys, h_values, _root, arrayLength, rootIndex, arrayLenPowerOf2 / 4
                );
            }
            else
            {
                fillBitonicTreeKeyValue<MIN_VAL>(
                    h_keys, h_values, _root, arrayLength, rootIndex, arrayLenPowerOf2 / 4
                );
            }
        }

        if (arrayLength == arrayLenPowerOf2)
        {
            _spare->key = h_keys[arrayLength - 1];
            _spare->value = sortingKeyOnly ? arrayLength - 1 : h_values[arrayLength - 1];
        }
        else
        {
            _spare->key = _sortOrder == ORDER_ASC ? MAX_VAL : MIN_VAL;
            _spare->value = arrayLenPowerOf2 - 1;
        }
    }

    /*
    Converts bitonic tree to array of keys. Doesn't put value of spare node into array.
    Not to be called directly - bottom function calls it.
    */
    void bitonicTreeToArrayKeyOnly(data_t *h_keys, node_t *node, uint_t arrayLength, uint_t arrayIndex, uint_t stride)
    {
        if (arrayIndex < arrayLength)
        {
            h_keys[arrayIndex] = node->key;
        }

        if (stride == 0)
        {
            return;
        }

        bitonicTreeToArrayKeyOnly(h_keys, node->left, arrayLength, arrayIndex - stride, stride / 2);
        if (arrayIndex < arrayLength)
        {
            bitonicTreeToArrayKeyOnly(h_keys, node->right, arrayLength, arrayIndex + stride, stride / 2);
        }
    }

    /*
    Converts bitonic tree to array of keys and values. Doesn't put value of spare node into array.
    Not to be called directly - bottom function calls it.
    */
    void bitonicTreeToArrayKeyValue(
        data_t *h_keys, data_t *h_values, node_t *node, uint_t arrayLength, uint_t arrayIndex, uint_t stride
    )
    {
        if (arrayIndex < arrayLength)
        {
            h_keys[arrayIndex] = node->key;
            h_values[arrayIndex] = node->value;
        }

        if (stride == 0)
        {
            return;
        }

        bitonicTreeToArrayKeyValue(h_keys, h_values, node->left, arrayLength, arrayIndex - stride, stride / 2);
        if (arrayIndex < arrayLength)
        {
            bitonicTreeToArrayKeyValue(h_keys, h_values, node->right, arrayLength, arrayIndex + stride, stride / 2);
        }
    }

    /*
    Copies data from device to host. If sorting keys only, than "h_values" contains NULL.
    */
    virtual void memoryCopyAfterSort(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        SortSequential::memoryCopyAfterSort(h_keys, h_values, arrayLength);
        bool sortingKeyOnly = h_values == NULL;

        if (arrayLength == 1)
        {
            h_keys[0] = _root->key;
            if (!sortingKeyOnly)
            {
                h_values[0] = _root->value;
            }
            return;
        }

        uint_t arrayLenPower2 = nextPowerOf2(arrayLength);

        if (sortingKeyOnly)
        {
            bitonicTreeToArrayKeyOnly(h_keys, _root, arrayLength, arrayLenPower2 / 2 - 1, arrayLenPower2 / 4);
        }
        else
        {
            bitonicTreeToArrayKeyValue(
                h_keys, h_values, _root, arrayLength, arrayLenPower2 / 2 - 1, arrayLenPower2 / 4
            );
        }

        // If array was padded, then there is no need to insert a spare node in last element of array.
        if (arrayLength < arrayLenPower2)
        {
            return;
        }

        // Inserts spare node
        h_keys[arrayLength - 1] = _spare->key;
        if (!sortingKeyOnly)
        {
            h_values[arrayLength - 1] = _spare->value;
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
    Executes adaptive bitonic sort on provided bitonic tree. Requires root node of bitonic tree, spare node
    (at beginning this is node with last array element with no children and parents) and sort order.
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
    Wrapper for sequential adaptive bitonic sort method.
    The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    */
    void sortKeyOnly()
    {
        if (_sortOrder == ORDER_ASC)
        {
            bitonicSortAdaptiveSequential<ORDER_ASC>(_root, _spare);
        }
        else
        {
            bitonicSortAdaptiveSequential<ORDER_DESC>(_root, _spare);
        }
    }

    /*
    Wrapper for sequential adaptive bitonic sort method.
    */
    void sortKeyValue()
    {
        sortKeyOnly();
    }

public:
    std::string getSortName()
    {
        return this->_sortName;
    }

    /*
    Method for destroying memory needed for sort. For sort testing purposes this method is public.
    */
    void memoryDestroy()
    {
        if (_arrayLength == 0)
        {
            return;
        }

        SortSequential::memoryDestroy();

        deleteBitonicTree(_root);
        delete _spare;
    }
};

#endif
