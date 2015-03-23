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
    return CreateDirectory(folderName, NULL);
}

/*
Creates folder if it doesn't already exist.
*/
bool createFolder(std::string folderName)
{
    return CreateDirectory(folderName.c_str(), NULL);
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
Reads array from file.
*/
void readArrayFromFile(std::string fileName, data_t *keys, uint_t arrayLength)
{
    readArrayFromFile((char*)fileName.c_str(), keys, arrayLength);
}

/*
Saves array to file.
*/
void writeArrayToFile(char *fileName, data_t *keys, uint_t arrayLength)
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

/*
Saves array to file.
*/
void writeArrayToFile(std::string fileName, data_t *keys, uint_t arrayLength)
{
    writeArrayToFile((char*)fileName.c_str(), keys, arrayLength);
}


/*
Appends provided text to file.
*/
void appendToFile(std::string fileName, std::string text)
{
    std::ofstream file;
    file.open(fileName, std::fstream::app);

    file << text;
    file.close();
}
