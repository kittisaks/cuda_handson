#include "bin_reader.h"
#include <iostream>

using namespace std;

int main(int argc, char ** argv) {

    BinInfo bi;
    int * arr = new int[800];
    
    if (binWriteArray<int>("test.bin", NULL, arr, 800))
        cout << "Error!" << endl;

    int32_t * arr2;
    if (binReadAsArray<int32_t>("test.bin", NULL, &arr2))
        cout << "Error2!" << endl;

    binDiscardArray(arr2);

    return 0;
}

