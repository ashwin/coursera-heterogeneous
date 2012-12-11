// GenDataMP2.cpp: Generate data for assignment MP2

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

float genRandomFloat()
{
    return ( (float) rand() / RAND_MAX );
}

void genMatrix( float* mat, int rows, int cols )
{
    for ( int r = 0; r < rows; ++r )
        for ( int c = 0; c < cols; ++c )
            mat[ cols * r + c ] = genRandomFloat();
}

void mulMatrices
(
const float* matA,
const float* matB,
float*       matC,
int ARows,
int ACols,
int BCols
)
{
    const int CRows = ARows;
    const int CCols = BCols;

    for ( int r = 0; r < CRows; ++r )
    {
        for ( int c = 0; c < CCols; ++c )
        {
            float sum = 0.0;

            for ( int z = 0; z < ACols; ++z )
                sum += matA[ ACols * r + z ] * matB[ BCols * z + c ];

            matC[ CCols * r + c ] = sum;
        }
    }
}

void writeMatrix( const float* mat, int rows, int cols, const char* fname )
{
    std::ofstream outFile( fname );

    if ( !outFile )
    {
        std::cout << "Error! Opening file: " << fname << " for writing matrix\n";
        exit(1);
    }

    std::cout << "Writing matrix to file: " << fname << std::endl;

    outFile << rows << std::endl;
    outFile << cols << std::endl;

    int idx = 0;

    for ( int r = 0; r < rows; ++r )
    {
        for ( int c = 0; c < cols; ++c )
        {
            outFile << mat[ idx++ ] << " ";
        }

        outFile << std::endl;
    }
}

int main( int argc, const char** argv )
{
    // Info for user

    std::cout << "GenDataMP2: Generates data files to use as input for assignment MP2.\n";
    std::cout << "Invoke as: GenDataMP2 [MatrixARows] [MatrixAColumns] [MatrixBColumns]\n\n";

    std::cout << "Datasets used in online submission are ...\n";
    std::cout << "Dataset0:  64  64  64\n";
    std::cout << "Dataset1: 128  64 128\n";
    std::cout << "Dataset2: 100 128  56\n";
    std::cout << "Dataset3: 256 128 256\n";
    std::cout << "Dataset4:  32 128  32\n";
    std::cout << "Dataset5: 200 100 256\n\n";

    // Read input

    if ( 4 != argc )
    {
        std::cout << "Error! Wrong number of arguments to program.\n";
        return 0;
    }

    const int ARows = atoi( argv[1] );
    const int ACols = atoi( argv[2] );
    const int BRows = ACols;
    const int BCols = atoi( argv[3] );
    const int CRows = ARows;
    const int CCols = BCols;

    std::cout << "Dimensions of matrix A = [" << ARows << " x " << ACols << "]\n";
    std::cout << "Dimensions of matrix B = [" << BRows << " x " << BCols << "]\n";
    std::cout << "Dimensions of matrix C = [" << CRows << " x " << CCols << "]\n";

    // Memory for matrices

    float* matA = new float[ ARows * ACols ];
    float* matB = new float[ BRows * BCols ];
    float* matC = new float[ CRows * CCols ];

    // Create matrices

    genMatrix( matA, ARows, ACols );
    genMatrix( matB, BRows, BCols );
    mulMatrices( matA, matB, matC, ARows, ACols, BCols );

    // Write to files

    writeMatrix( matA, ARows, ACols, "matA.txt" );
    writeMatrix( matB, BRows, BCols, "matB.txt" );
    writeMatrix( matC, CRows, CCols, "matC.txt" );

    return 0;
}
