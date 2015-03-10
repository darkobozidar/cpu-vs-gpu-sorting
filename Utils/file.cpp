#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"


/*
Creates folder if it doesn't already exist.
*/
bool createFolder(char* folderName)
{
    return CreateDirectory(folderName, NULL);;
}

/*
Reads array from file.
*/
void readArrayFromFile(char *fileName, data_t *keys, uint_t arrayLength)
{
    std::ifstream file(fileName);
    std::string fileNumbers;
    std::getline(file, fileNumbers, '\n');
    std::stringstream numbersStream(fileNumbers);

    for (int i = 0; numbersStream.good() && i < arrayLength; i++)
    {
        numbersStream >> keys[i];
    }

    file.close();
}

/*
Saves array to file.
*/
void saveArrayToFile(char *fileName, data_t *keys, uint_t arrayLength)
{
    std::ofstream file(fileName);

    for (uint_t i = 0; i < arrayLength; i++)
    {
        file << keys[i];
        file << (i < arrayLength - 1 ? "\t" : "");
    }

    file << std::endl;
    file.close();
}
