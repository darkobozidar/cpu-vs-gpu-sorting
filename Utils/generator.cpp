#include <stdio.h>
#include <stdint.h>
#include <random>
#include <iostream>
#include <functional>
#include <chrono>

#include "data_types_common.h"

using namespace std;


void fillUniform(data_t *keys, uint_t tableLen, uint_t interval)
{
    auto seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    auto generator = std::bind(std::uniform_int_distribution<data_t>(0, interval), mt19937(seed));

    for (uint_t i = 0; i < tableLen; i++)
    {
        keys[i] = generator();
    }
}

/*
Fills keys with random numbers.
*/
void fillTableKeysOnly(data_t *keys, uint_t tableLen, uint_t interval, data_dist_t distribution)
{
    switch (distribution)
    {
        case DISTRIBUTION_UNIFORM:
            fillUniform(keys, tableLen, interval);
            break;

        default:
            printf("Invalid distribution parameter.\n");
            getchar();
            exit(EXIT_FAILURE);
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
