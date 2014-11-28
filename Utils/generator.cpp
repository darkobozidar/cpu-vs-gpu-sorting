#include <stdio.h>
#include <stdint.h>
#include <random>
#include <iostream>
#include <functional>
#include <chrono>

#include "data_types_common.h"

using namespace std;


/*
Fills keys with random numbers.
*/
void fillTableKeysOnly(data_t *keys, uint_t tableLen, uint_t interval, data_dist_t distribution)
{
    auto seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    // Choose appropriate random generator according to data type
    auto generator = std::bind(std::uniform_int_distribution<data_t>(0, interval), mt19937(seed));

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
                double sum;
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
            double value = generator();

            for (uint_t i = 0; i < tableLen; i++)
            {
                keys[i] = value;
            }
            break;
        }
        default:
        {
            printf("Invalid distribution parameter.\n");
            getchar();
            exit(EXIT_FAILURE);
        }
    }

    // TODO implement
    /*DISTRIBUTION_UNIFORM,
    DISTRIBUTION_GAUSSIAN,
    DISTRIBUTION_ZERO,
    DISTRIBUTION_BUCKET,
    DISTRIBUTION_STAGGERED,
    DISTRIBUTION_SORTED_ASC,
    DISTRIBUTION_SORTED_DESC*/
}

/*
Fills keys with random numbers and values with consectuive values (for stability test).
*/
void fillTableKeyValue(data_t *keys, data_t *values, uint_t tableLen, uint_t interval, data_dist_t distribution)
{
    fillTableKeysOnly(keys, tableLen, interval, distribution);

    for (uint_t i = 0; i < tableLen; i++)
    {
        values[i] = i;
    }
}
