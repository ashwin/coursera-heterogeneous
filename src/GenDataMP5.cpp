// GenDataMP5.cpp: Generate data for assignment MP5

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

typedef std::vector< float > FloatVec;

float genRandomFloat()
{
    return ( (float) rand() / RAND_MAX );
}

void genVector( FloatVec& vec, int vecLen )
{
    for ( int i = 0; i < vecLen; ++i )
        vec.push_back( genRandomFloat() );
}

void scanVector( const FloatVec& vecA, FloatVec& vecB )
{
    vecB.push_back(vecA[0]);
    int len = vecA.size();

    for (int i = 1; i < len; ++i) {
        vecB.push_back(vecB[i-1] + vecA[i]);
    }    
}

void writeVector( const FloatVec& vec, const char* fname )
{
    std::ofstream outFile( fname );

    if ( !outFile )
    {
        std::cout << "Error! Opening file: " << fname << " for writing vector.\n";
        exit(1);
    }

    std::cout << "Writing vector to file: " << fname << std::endl;

    const int vecLen = (int) vec.size();

    outFile << vecLen << std::endl;

    for ( int i = 0; i < vecLen; ++i )
        outFile << vec[i] << std::endl;
}

int main( int argc, const char** argv )
{
    // Info for user

    std::cout << "GenDataMP5: Generates data files to use as input for assignment MP5.\n";
    std::cout << "Invoke as: GenDataMP5 [VectorLength]\n\n";

    // Read input

    if ( 2 != argc )
    {
        std::cout << "Error! Wrong number of arguments to program.\n";
        return 0;
    }

    // Create vectors

    const int vecLen = atoi( argv[1] );

    FloatVec vecA;
    FloatVec vecB;

    genVector( vecA, vecLen );
    scanVector( vecA, vecB );

    // Write to files

    writeVector( vecA, "vecA.txt" );
    writeVector( vecB, "vecB.txt" );

    return 0;
}
