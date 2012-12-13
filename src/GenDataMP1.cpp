// GenDataMP1.cpp: Generate data for assignment MP1

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

void addVector( const FloatVec& vecA, const FloatVec& vecB, FloatVec& vecC )
{
    assert( vecA.size() == vecB.size() );

    for ( int i = 0; i < (int) vecA.size(); ++i )
        vecC.push_back( vecA[i] + vecB[i] );
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

    std::cout << "GenDataMP1: Generates data files to use as input for assignment MP1.\n";
    std::cout << "Invoke as: GenDataMP1 [VectorLength]\n\n";

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
    FloatVec vecC;

    genVector( vecA, vecLen );
    genVector( vecB, vecLen );
    addVector( vecA, vecB, vecC );

    // Write to files

    writeVector( vecA, "vecA.txt" );
    writeVector( vecB, "vecB.txt" );
    writeVector( vecC, "vecC.txt" );

    return 0;
}