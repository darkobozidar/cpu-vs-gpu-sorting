#ifndef DATA_TYPES_H
#define DATA_TYPES_H


typedef enum TransferDirection direct_t;

/*
Enum used to denote, in which direction is the data transfered to during local and global quicksort.
*/
enum TransferDirection
{
    PRIMARY_MEM_TO_BUFFER,
    BUFFER_TO_PRIMARY_MEM
};

#endif
