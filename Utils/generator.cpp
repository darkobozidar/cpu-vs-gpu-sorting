#include <stdio.h>
#include <stdint.h>
#include <random>
#include <iostream>
#include <functional>
#include <chrono>
#include <stdint.h>

#include "data_types_common.h"
#include "cuda.h"
#include "sort_correct.h"

using namespace std;

// Needed to ensure different seed every time
uint_t generatorCalls = 0;


/*
Fills keys with random numbers.
TODO break into separate function for every distribution (currently having difficulties due to "auto" data type).
*/
void fillArrayKeyOnly(data_t *keys, uint_t tableLen, uint_t interval, uint_t bucketSize, data_dist_t distribution)
{
    auto seed = chrono::high_resolution_clock::now().time_since_epoch().count() + generatorCalls++;
#if DATA_TYPE_BITS == 32
    auto generator = std::bind(std::uniform_int_distribution<data_t>(0, interval), mt19937(seed));
#else
    auto generator = std::bind(std::uniform_int_distribution<data_t>(0, interval), mt19937_64(seed));
#endif

    switch (distribution)
    {
        case DISTRIBUTION_UNIFORM:
        {
            for (uint_t i = 0; i < tableLen; i++)
            {
                keys[i] = generator();
            }

            break;
        }
        case DISTRIBUTION_GAUSSIAN:
        {
            double numValues = 4;  // How many values are used for average when generating random numbers

            for (uint_t i = 0; i < tableLen; i++)
            {
                data_t sum = 0;

                for (uint_t j = 0; j < numValues; j++)
                {
                    sum += generator();
                }

                keys[i] = sum / numValues;
            }

            break;
        }
        case DISTRIBUTION_ZERO:
        {
            data_t value = generator();

            for (uint_t i = 0; i < tableLen; i++)
            {
                keys[i] = value;
            }

            break;
        }
        case DISTRIBUTION_BUCKET:
        {
            uint_t index = 0;
            data_t bucketIncrement = (MAX_VAL / bucketSize + 1);

            // Fills the buckets
            for (uint_t i = 0; i < bucketSize; i++)
            {
                for (uint_t j = 0; j < bucketSize; j++)
                {
                    for (uint_t k = 0; k < tableLen / bucketSize / bucketSize; k++)
                    {
                        keys[index++] = (data_t)(j * bucketIncrement + (generator() >> bucketSize));
                    }
                }
            }

            // Fills the rest of the data into table
            for (; index < tableLen; index++)
            {
                keys[index] = generator();
            }

            break;
        }
        case DISTRIBUTION_STAGGERED:
        {
            uint_t index = 0;

            for (uint_t i = 0; i < bucketSize; i++)
            {
                uint_t j;
                data_t bucketIncrement;

                if (i < (bucketSize / 2))
                {
                    bucketIncrement = 2 * bucketSize + 1;
                }
                else
                {
                    bucketIncrement = (i - (bucketSize / 2)) * 2;
                }

                bucketIncrement = bucketIncrement * ((MAX_VAL / bucketSize) + 1);

                for (j = 0; j < tableLen / bucketSize; j++)
                {
                    keys[index++] = bucketIncrement + ((generator()) / bucketSize) + 1;
                }
            }

            for (; index < tableLen; index++)
            {
                keys[index] = generator();
            }

            break;
        }
        case DISTRIBUTION_SORTED_ASC:
        {
            for (uint_t i = 0; i < tableLen; i++)
            {
                keys[i] = generator();
            }
            sortCorrect(keys, tableLen, ORDER_ASC);
            break;
        }
        case DISTRIBUTION_SORTED_DESC:
        {
            for (uint_t i = 0; i < tableLen; i++)
            {
                keys[i] = generator();
            }
            sortCorrect(keys, tableLen, ORDER_DESC);
            break;
        }
        default:
        {
            printf("Invalid distribution parameter.\n");
            getchar();
            exit(EXIT_FAILURE);
        }
    }
}

/*
Fills keys with random values on provided interval.
*/
void fillArrayKeyOnly(data_t *keys, uint_t tableLen, uint_t interval, data_dist_t distribution)
{
    fillArrayKeyOnly(keys, tableLen, interval, getMaxThreadsPerBlock(), distribution);
}

/*
Fills array with sequential (consequently unique) values.
*/
void fillArrayValueOnly(data_t *values, uint_t tableLen)
{
    for (uint_t i = 0; i < tableLen; i++)
    {
        values[i] = i;
    }
}

/*
Fills keys with random numbers and values with consecutive values (for stability test).
*/
void fillArrayKeyValue(data_t *keys, data_t *values, uint_t tableLen, uint_t interval, data_dist_t distribution)
{
    fillArrayKeyOnly(keys, tableLen, interval, distribution);
    fillArrayValueOnly(values, tableLen);
}
