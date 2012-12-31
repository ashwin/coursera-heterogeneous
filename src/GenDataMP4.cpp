// GenDataMP4.cpp: Generate data for assignment MP4

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

void sumVector( const FloatVec& vecA, FloatVec& vecB )
{
    float sum = 0.0;
    for ( int i = 0; i < (int) vecA.size(); ++i )
        sum += vecA[i];

    vecB.push_back( sum);
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

    std::cout << "GenDataMP4: Generates data files to use as input for assignment MP4.\n";
    std::cout << "Invoke as: GenDataMP4 [VectorLength]\n\n";

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
    sumVector( vecA, vecB );

    // Write to files

    writeVector( vecA, "vecA.txt" );
    writeVector( vecB, "vecB.txt" );

    return 0;
}
