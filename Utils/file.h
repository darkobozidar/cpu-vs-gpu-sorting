#ifndef FILE_H
#define FILE_H

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"


bool createFolder(char* folderName);
bool createFolder(std::string folderName);
void readArrayFromFile(char *fileName, data_t *keys, uint_t arrayLength);
void saveArrayToFile(char *fileName, data_t *keys, uint_t arrayLength);

#endif
