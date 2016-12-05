#ifndef _BIN_READER_H
#define _BIN_READER_H

#include <stdint.h>
#include <fstream>
#include <typeinfo>

#define MAGIC 0x1b5e0dba00000000

typedef struct {
    uint64_t magic;
    uint64_t offset;
    uint64_t size;
    uint64_t type;
} BinHeader;

typedef enum {
    BinTypeInvalid = 0x00,
    BinTypeUint32,
    BinTypeUint64,
    BinTypeInt32,
    BinTypeInt64,
    BinTypeFloat, 
    BinTypeDouble,
    BinTypeFirst = BinTypeInvalid,
    BinTypeLast = BinTypeDouble
} BinType;

typedef struct {
    std::fstream * handle;
    uint64_t       offset;
    uint64_t       size;
    BinType        type;
} BinInfo;

#define CLEAN_AND_RETURN(inst, ret) do {delete inst; return ret;} while(0)

int binOpenForRead(const char * filename, BinInfo * bi) {


    std::fstream * handle = new std::fstream();
    handle->open(filename, std::ios::in | std::ios::binary | std::ios::ate);

    if (!handle->is_open())
        CLEAN_AND_RETURN(handle, -1);

    uint64_t size = handle->tellg();
    if (size < sizeof(BinHeader))
        CLEAN_AND_RETURN(handle, -1);

    handle->seekg(std::ios::beg);

    BinHeader header;
    handle->read((char *) &header, sizeof(BinHeader));

    if (header.magic != MAGIC)
        CLEAN_AND_RETURN(handle, -1);

    if ((header.type < BinTypeFirst) || (header.type > BinTypeLast))
        CLEAN_AND_RETURN(handle, -1);

    bi->handle = handle;
    bi->offset = header.offset;
    bi->size   = size;
    bi->type   = static_cast<BinType>(header.type);

    return 0;
}

int binOpenForWrite(const char * filename, BinInfo * bi) {

    std::fstream * handle = new std::fstream();
    handle->open(filename, std::ios::out | std::ios::binary);

    if (!handle->is_open())
        CLEAN_AND_RETURN(handle, -1);

    bi->handle = handle;
    bi->offset = 0;
    bi->size   = 0;
    bi->type   = BinTypeInvalid;

    return 0;
}

int binClose(BinInfo & bi) {

    if (bi.handle == NULL)
        return -1;

    if (!bi.handle->is_open())
        return 0;

    bi.handle->close();

    return 0;
}

int CheckBinTypeCompatibility(const std::type_info & t, BinType * type)
{
#define CHECK_AND_RETURN(typ, binType, ret) \
    if (t==typeid(typ)) {*type=binType; return ret;} while(0)

    if ((sizeof(unsigned int) != sizeof(uint32_t)) ||
        (sizeof(unsigned long) != sizeof(uint64_t)) ||
        (sizeof(int) != sizeof(int32_t)) ||
        (sizeof(long) != sizeof(int64_t)))
        return -1;

    CHECK_AND_RETURN(uint32_t, BinTypeUint32, 0);
    CHECK_AND_RETURN(unsigned int, BinTypeUint32, 0);
    CHECK_AND_RETURN(uint64_t, BinTypeUint64, 0);
    CHECK_AND_RETURN(unsigned long, BinTypeUint64, 0);
    CHECK_AND_RETURN(int32_t, BinTypeInt32, 0);
    CHECK_AND_RETURN(int, BinTypeInt32, 0);
    CHECK_AND_RETURN(int64_t, BinTypeInt64, 0);
    CHECK_AND_RETURN(long, BinTypeInt64, 0);
    CHECK_AND_RETURN(float, BinTypeFloat, 0);
    CHECK_AND_RETURN(double, BinTypeDouble, 0);

    return -1;
        
#undef CHECK_AND_RETURN    
}

template <typename T> int binReadAsArray(
    const char * filename, BinInfo * bi, T ** arr) {

    if (arr == NULL)
        return -1;

    BinInfo ibi;

    int ret;
    ret = binOpenForRead(filename, &ibi);
    if (ret != 0)
        return -1;

    BinType type;
    if (CheckBinTypeCompatibility(typeid(T), &type))
        return -1;

    if (ibi.type != type)
        return -1;

    uint64_t count = ibi.size / sizeof(T);
    T * iarr = new T [count];
    ibi.handle->seekg(ibi.offset, std::ios::beg);
    ibi.handle->read((char *) iarr, ibi.size);

    *arr = iarr;
    if (bi != NULL)
        *bi = ibi;

    ret = binClose(ibi);
    if (ret != 0)
        return -1;

    return 0;
}

template <typename T> int binDiscardArray(T * arr) {

    delete arr;

    return 0;
}

template <typename T> int binWriteArray(
    const char * filename, BinInfo * bi, T * arr, size_t size) {

    if (arr == NULL)
        return -1;

    BinInfo ibi;

    int ret;
    ret = binOpenForWrite(filename, &ibi);
    if (ret != 0)
        return -1;

    BinType type;
    if (CheckBinTypeCompatibility(typeid(T), &type))
        return -1;

    ibi.offset = sizeof(BinHeader);
    ibi.size   = size * sizeof(T);
    ibi.type   = type;

    BinHeader header;
    header.magic  = MAGIC;
    header.offset = ibi.offset;
    header.type   = ibi.type;

    ibi.handle->write((char *) &header, sizeof(BinHeader));
    ibi.handle->write((char *) arr, ibi.size);

    if (bi != NULL)
        *bi = ibi;

    ret = binClose(ibi);
    if (ret != 0)
        return -1;

    return 0;
}

#endif //BIN_READER_H

