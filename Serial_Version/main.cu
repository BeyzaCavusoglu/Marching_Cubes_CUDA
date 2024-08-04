#include <math.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cuda.h>
#include <iostream>
#include "Data.h"
#include "helper_math.h"
#include "helper_cuda.h"
#include "IO.h"

using namespace std;

///////////////////////////////////////////////////////////////////
//    You don't need to change anything inside this namespace    //
//       (Unless you are sure about what you are doing)          //
///////////////////////////////////////////////////////////////////
namespace
{
    void showMemUsage()
    {
        // show memory usage of GPU
        size_t free_byte;
        size_t total_byte;

        cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

        if (cudaSuccess != cuda_status)
        {
            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
            exit(1);
        }

        double free_db = (double)free_byte;
        double total_db = (double)total_byte;
        double used_db = total_db - free_db;
        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
               used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
    }

    // initialize the sdf of the desired shape
    __host__ __device__ float linkSdf(float3 const &pos)
    {
        float radBig = 0.5;
        float radSmall = 0.2;
        float le = 0.3;
        float3 q = make_float3(pos.x, max(abs(pos.y) - le, 0.0), pos.z);
        // return Norm(Norm(q.x, q.y) - radBig, q.z) - radSmall;
        float fnorm = sqrt(q.x * q.x + q.y * q.y) - radBig;
        return sqrt(fnorm * fnorm + q.z * q.z) - radSmall;
    };

    // initialize the twist transformation function
    __host__ __device__ float opTwist(float3 const &pos, double twist)
    {
        float c = cosf(twist * pos.y);
        float s = sinf(twist * pos.y);
        float3 q = make_float3(c * pos.z - s * pos.x, pos.y, s * pos.z + c * pos.x);
        return linkSdf(q);
    };

    // function generating points based on the given geometry
    __host__ __device__ float3 NumGrad(float3 const &p, float twist)
    {
        const float Eps = 1e-6;
        float3 epsX = make_float3(Eps, 0, 0);
        float3 epsY = make_float3(0, Eps, 0);
        float3 epsZ = make_float3(0, 0, Eps);
        float gx = (opTwist(p + epsX, twist) - opTwist(p - epsX, twist)) / 2;
        float gy = (opTwist(p + epsY, twist) - opTwist(p - epsY, twist)) / 2;
        float gz = (opTwist(p + epsZ, twist) - opTwist(p - epsZ, twist)) / 2;
        return make_float3(gx, gy, gz);
    }

    __host__ __device__ float3 LerpVertex(float isoLevel, int i1, int i2, float3 *pos, float *sdf)
    {
        float const Eps = 1e-5;
        float const v1 = sdf[i1];
        float const v2 = sdf[i2];
        float3 const &p1 = pos[i1];
        float3 const &p2 = pos[i2];

        if (abs(isoLevel - v1) < Eps)
            return p1;
        if (abs(isoLevel - v2) < Eps)
            return p2;
        if (abs(v1 - v2) < Eps)
            return p1;

        float mu = (isoLevel - v1) / (v2 - v1);
        return p1 + ((p2 - p1) * mu);
    }

    __host__ __device__ int SignConfig(float isoLevel, float *sdf)
    {
        int edgeIndex = 0;

        for (int i = 0; i < 8; ++i)
        {
            if (sdf[i] < isoLevel)
            {
                edgeIndex |= 1 << i;
            }
        }

        return edgeIndex;
    }

    __host__ __device__ int Intersect(float3 *pos, float *sdf, float3 *intersect, float iso)
    {
        // idea:
        // from signs at 8 corners of cube a sign configuration (256 possible ones) is computed
        // this configuration can be used to index into a table that tells which of the 12 edges are intersected
        // find vertices adjacent to edges and interpolate cut vertex and store it in IntersectionInfo object

        int signConfig = SignConfig(iso, sdf);

        for (int e = 0; e < 12; ++e)
        {
            if (signConfigToIntersectedEdges(signConfig) & edges(e).edgeFlag)
            {
                int v0 = edges(e).vert0;
                int v1 = edges(e).vert1;
                float3 vert = LerpVertex(iso, v0, v1, pos, sdf);
                intersect[e] = vert;
            }
        }

        return signConfig;
    }

}

/// Given a grid cube and an isolevel the triangles (5 max)
/// required to represent the isosurface in the cube are computed.
///////////////////////////////////////////////////////////////////
//      You don't HAVE TO change this function, but you can      //
///////////////////////////////////////////////////////////////////
__host__ __device__ void Triangulate(
    float twist,
    float3 *meshVertices,
    float3 *meshNormals,
    int signConfig,
    float3 *intersect)
{
    // Cube is entirely in/out of the surface. Generate no triangles.
    if (signConfig == 0 || signConfig == 255)
        return;

    // consult with the table to see how the triangles will be generated
    const int *tri = signConfigToTriangles(signConfig);

    // there can be at most 15 vertices, but it varies case by case
    for (int i = 0; tri[i] != -1; i += 3)
    {
        float3 v0 = intersect[tri[i]];
        float3 v1 = intersect[tri[i + 1]];
        float3 v2 = intersect[tri[i + 2]];

        float3 normal0 = normalize(NumGrad(v0, twist));
        float3 normal1 = normalize(NumGrad(v1, twist));
        float3 normal2 = normalize(NumGrad(v2, twist));

        meshVertices[i] = v0;
        meshVertices[i + 1] = v1;
        meshVertices[i + 2] = v2;

        meshNormals[i] = normal0;
        meshNormals[i + 1] = normal1;
        meshNormals[i + 2] = normal2;
    }
}

///////////////////////////////////////////////////////////////////
//             This is the baseline, DON'T change it             //
///////////////////////////////////////////////////////////////////
void MarchCube(
    Rect3 const &domain,
    float3 const &cubeSize,
    float twist,
    float isoLevel,
    float3 *meshVertices,
    float3 *meshNormals)
{
    int NumX = static_cast<int>(ceil(domain.size.x / cubeSize.x));
    int NumY = static_cast<int>(ceil(domain.size.y / cubeSize.y));
    int NumZ = static_cast<int>(ceil(domain.size.z / cubeSize.z));

    float3 *intersect = new float3[12];
    // iterate over each cube in the mesh
    for (int ix = 0; ix < NumX; ++ix)
    {
        float x = domain.min.x + ix * cubeSize.x;

        for (int iy = 0; iy < NumY; ++iy)
        {
            float y = domain.min.y + iy * cubeSize.y;

            for (int iz = 0; iz < NumZ; ++iz)
            {
                float z = domain.min.z + iz * cubeSize.z;
                float3 min = make_float3(x, y, z);

                // create a cube made of 8 vertices
                float3 pos[8];
                float sdf[8];

                Rect3 space = {min, cubeSize};

                float mx = space.min.x;
                float my = space.min.y;
                float mz = space.min.z;

                float sx = space.size.x;
                float sy = space.size.y;
                float sz = space.size.z;

                pos[0] = space.min;
                pos[1] = make_float3(mx + sx, my, mz);
                pos[2] = make_float3(mx + sx, my, mz + sz);
                pos[3] = make_float3(mx, my, mz + sz);
                pos[4] = make_float3(mx, my + sy, mz);
                pos[5] = make_float3(mx + sx, my + sy, mz);
                pos[6] = make_float3(mx + sx, my + sy, mz + sz);
                pos[7] = make_float3(mx, my + sy, mz + sz);

                // fill in the vertices of the cube
                for (int i = 0; i < 8; ++i)
                {
                    float sd = opTwist(pos[i], twist);
                    if (sd == 0)
                        sd += 1e-6;
                    sdf[i] = sd;
                }

                // map the vertices under the isosurface to intersecting edges
                int signConfig = Intersect(pos, sdf, intersect, isoLevel);

                // now create and store the triangle data
                int offset = (NumZ * NumY * ix + NumZ * iy + iz) * 16;
                Triangulate(twist, meshVertices + offset, meshNormals + offset, signConfig, intersect);
            }
        }
    }
}

///////////////////////////////////////////////////////////////////
//                             PART 1&2                          //
//         Doing PART 2 means you get full points for 1&2        //
//                You can add additional parameters              //
///////////////////////////////////////////////////////////////////
__global__ void MarchCubeCUDA(
    Rect3 *domainP,
    float3 *cubeSizeP,
    float twist,
    float isoLevel,
    float3 *meshVertices,
    float3 *meshNormals)
{
    
}

///////////////////////////////////////////////////////////////////
//                             PART 3                            //
//                You can add additional parameters              //
///////////////////////////////////////////////////////////////////
__global__ void MarchCubeCUDAMultiframe(
    Rect3 *domainP,
    float3 *cubeSizeP,
    int frameNum,
    float maxTwist,
    float isoLevel,
    float3 *meshVertices,
    float3 *meshNormals)
{

}

///////////////////////////////////////////////////////////////////
//                             PART 4                            //
//                You can add a function if needed               //
///////////////////////////////////////////////////////////////////
// __global__ void MarchCubeCUDATwoPointers() {}


// Correctness test. May not work well for you, so test the generated shapes instead
void TestCorrectness(int frameSize, float3 *result, float3 *truth, int frame)
{
    float eps = 1e-4;
    int wrong = 0;
    for (int i = 0; i < frameSize; i++)
    {
        if (abs(result[i].x - truth[i].x) > eps ||
            abs(result[i].y - truth[i].y) > eps ||
            abs(result[i].z - truth[i].z) > eps)
        {
            printf("Vertex under the index %d/%d: \ngot: [%f,%f,%f] \nexpected [%f,%f,%f]\n",
                   i, frameSize, result[i].x, result[i].y, result[i].z, truth[i].x, truth[i].y, truth[i].z);
            wrong++;
            if (wrong > frameSize / 20)
            {
                printf("Test for frame %d failed \n", frame);
                return;
            }
        }
    }
    printf("Test for frame %d passed! \n", frame);
}

int main(int argc, char *argv[])
{
    int cubesRes = 50;
    int frameNum = 10;
    int saveObj = 0;
    int correctTest = 0;
    int numBlocks = 40;
    int numThreads = 64;
    int part = -1;

    opterr = 0;
    int c;
    int index;

    // read the command-line args
    while ((c = getopt(argc, argv, "con:p:f:b:t:")) != -1)
    {
        switch (c)
        {
        case 'c':
            correctTest = 1;
            break;
        case 'o':
            saveObj = 1;
            break;
        case 'n':
            cubesRes = stoi(optarg);
            break;
        case 'p':
            part = stoi(optarg);
            break;
        case 'b':
            numBlocks = stoi(optarg);
            break;
        case 't':
            numThreads = stoi(optarg);
            break;
        case 'f':
            frameNum = stoi(optarg);
            break;
        case '?':
            if (optopt == 'n' || optopt == 'f')
                fprintf(stderr, "Option -%c requires an argument.\n", optopt);
            else if (isprint(optopt))
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
            exit(0);
        default:
            exit(0);
        }
    }

    for (index = optind; index < argc; index++)
        printf("Non-option argument %s\n", argv[index]);

    using std::chrono::duration;
    using std::chrono::high_resolution_clock;
    using std::chrono::time_point;

    printf("resolution (n) = %d, number of frames (f) = %d\nblocks (b) = %d, threads (t) = %d", cubesRes, frameNum, numBlocks, numThreads);

    float maxTwist = 5.0;
    float twist = 0.0;
    Rect3 domain;
    
    // the domain is from -1 to 1 in each axis
    domain.min = make_float3(-1.0, -1.0, -1.0);
    domain.size = make_float3(2.0, 2.0, 2.0);
    // set the size of each cube based on the -n parameter
    float3 cubeSize = domain.size * (1.0 / cubesRes);
    int frame;

    // set the number of cubes in each axis 
    int NumX = static_cast<int>(ceil(domain.size.x / cubeSize.x));
    int NumY = static_cast<int>(ceil(domain.size.y / cubeSize.y));
    int NumZ = static_cast<int>(ceil(domain.size.z / cubeSize.z));
    // the total (potential maximum) number of vertices in a frame
    int frameSize = NumX * NumY * NumZ * 16;

    ///////////////////////////////////////////////////////////////////
    //         Allocate required memory and buffers here             //
    ///////////////////////////////////////////////////////////////////
    
    ///////////////////////////////////////////////////////////////////
    //         Do not change the memory allocated for testing        //
    ///////////////////////////////////////////////////////////////////

    // this is for copying back from the GPU
    float3 *meshVertices_h;
    checkCudaErrors(cudaMallocHost(&meshVertices_h, frameSize * frameNum * sizeof(float3)));
    float3 *meshNormals_h;
    checkCudaErrors(cudaMallocHost(&meshNormals_h, frameSize * frameNum * sizeof(float3)));

    // this is for accuracy tests
    float3 *meshVertices_test;
    checkCudaErrors(cudaMallocHost(&meshVertices_test, frameSize * frameNum * sizeof(float3)));
    float3 *meshNormals_test;
    checkCudaErrors(cudaMallocHost(&meshNormals_test, frameSize * frameNum * sizeof(float3)));

    time_point<high_resolution_clock> start;
    time_point<high_resolution_clock> end;
    double totalTime;
    // for testing purposes
    if (correctTest || part < 1)
    {
        int offset = 0;
        start = high_resolution_clock::now();
        for (frame = 0; frame < frameNum; frame++)
        {
            MarchCube(domain, cubeSize, twist, 0, meshVertices_test + offset, meshNormals_test + offset);

            if (saveObj)
            {
                string filename = "cpu_link_f" + to_string(frame) + "_n" + to_string(cubesRes) + ".obj";
                WriteObjFile(frameSize, meshVertices_test + offset, meshNormals_test + offset, filename);
            }
            
            offset += frameSize;
            twist += 1.0 / float(frameNum) * maxTwist;
        }
        end = high_resolution_clock::now();
        twist = 0;
        totalTime = (duration<double>(end - start)).count();

        printf("\nCPU\nTime taken: %f | Performance: %f mln_cubes/sec \n",
               totalTime, (double(frameNum * NumX * NumY * NumZ) / 1000000.0 / totalTime));
    }

    double kernelTime = 0;
    double memcpyTime = 0;
    double extraTime = 0;
    ///////////////////////////////////////////////////////////////////
    //                        PART 1&2 RUN:                          //
    ///////////////////////////////////////////////////////////////////
    if (part == -1 || part == 1 || part == 2)
    {
        for (frame = 0; frame < frameNum; frame++)
        {
            start = high_resolution_clock::now();
            ///////////////////////////////////////////////////////////
            //                   Launch the kernel                   //
            ///////////////////////////////////////////////////////////
            // MarchCubeCUDA<<<numBlocks, numThreads>>>(domain_d, cubeSize_d, twist, 0, meshVertices_d, meshNormals_d);
            checkCudaErrors(cudaDeviceSynchronize());
            end = high_resolution_clock::now();
            kernelTime += (duration<double>(end - start)).count();

            start = high_resolution_clock::now();
            ///////////////////////////////////////////////////////////
            //            Copy the result back to host               //
            ///////////////////////////////////////////////////////////
            end = high_resolution_clock::now();

            memcpyTime += (duration<double>(end - start)).count();

            start = high_resolution_clock::now();
            // save the object file if told so
            if (saveObj)
            {
                string filename = "part_1and2_link_f" + to_string(frame) + "_n" + to_string(cubesRes) + ".obj";
                WriteObjFile(frameSize, meshVertices_h, meshNormals_h, filename);
            }

            // testing
            if (correctTest)
            {
                TestCorrectness(frameSize, meshVertices_h, meshNormals_h, frame);
            }
            end = high_resolution_clock::now();
            extraTime += (duration<double>(end - start)).count();

            twist += 1.0 / float(frameNum) * maxTwist;
        }

        printf("\nPart1&2\nTime taken: \nKernels: %f sec | Performance: %f mln_cubes/sec \nMemcpy:  "
               "%f sec \nExtra:   %f sec \nTotal:   %f sec\n",
               kernelTime, (double(frameNum * NumX * NumY * NumZ) / 1000000.0 / kernelTime),
               memcpyTime, extraTime, (kernelTime + memcpyTime + extraTime));
        showMemUsage();
        cudaMemset(meshNormals_h, 0, frameNum * frameSize * sizeof(float3));
        cudaMemset(meshVertices_h, 0, frameNum * frameSize * sizeof(float3));
        twist = 0;
    }

    ///////////////////////////////////////////////////////////////////
    //   Re-allocate some of the memory and buffers here if needed   //
    ///////////////////////////////////////////////////////////////////
    //                         PART 3 RUN:                           //
    ///////////////////////////////////////////////////////////////////
    if (part == -1 || part == 3)
    {
        start = high_resolution_clock::now();
        ///////////////////////////////////////////////////////////////
        //                     Launch the kernel                     //
        ///////////////////////////////////////////////////////////////
        // MarchCubeCUDAMultiframe<<<numBlocks, numThreads>>>(domain_d, cubeSize_d, frameNum, maxTwist, 0, meshVertices_d, meshNormals_d);
        checkCudaErrors(cudaDeviceSynchronize());
        end = high_resolution_clock::now();
        kernelTime = (duration<double>(end - start)).count();

        start = high_resolution_clock::now();
        ///////////////////////////////////////////////////////////////
        //              Copy the result back to host                 //
        ///////////////////////////////////////////////////////////////
        end = high_resolution_clock::now();
        memcpyTime = (duration<double>(end - start)).count();

        int offset = 0;
        start = high_resolution_clock::now();
        for (frame = 0; frame < frameNum; frame++)
        {
            if (saveObj)
            {
                string filename = "part_3_link_f" + to_string(frame) + "_n" + to_string(cubesRes) + ".obj";
                WriteObjFile(frameSize, meshVertices_h + offset, meshNormals_h + offset, filename);
            }

            if (correctTest)
            {
                TestCorrectness(frameSize, meshVertices_h + offset, meshNormals_h + offset, frame);
            }
            offset += frameSize;
        }
        end = high_resolution_clock::now();
        extraTime = (duration<double>(end - start)).count();

        printf("\nPart3\nTime taken: \nKernels: %f sec | Performance: %f mln_cubes/sec \nMemcpy:  "
               "%f sec \nExtra:   %f sec \nTotal:   %f sec\n",
               kernelTime, (double(frameNum * NumX * NumY * NumZ) / 1000000.0 / kernelTime),
               memcpyTime, extraTime, (kernelTime + memcpyTime + extraTime));
        showMemUsage();
        cudaMemset(meshNormals_h, 0, frameNum * frameSize * sizeof(float3));
        cudaMemset(meshVertices_h, 0, frameNum * frameSize * sizeof(float3));
        twist = 0;
    }

    ///////////////////////////////////////////////////////////////////
    //   Re-allocate some of the memory and buffers here if needed   //
    ///////////////////////////////////////////////////////////////////
    //                         PART 4 RUN:                           //
    ///////////////////////////////////////////////////////////////////
    if (part == -1 || part == 4)
    {
        int offset = 0;
        start = high_resolution_clock::now();
        for (frame = 0; frame < frameNum; frame++)
        {
            ///////////////////////////////////////////////////////////
            //                   Launch the kernel                   //
            ///////////////////////////////////////////////////////////
            //         Copy the result back to host (async)          //
            ///////////////////////////////////////////////////////////

            // save the object file if told so
            if (saveObj)
            {
                checkCudaErrors(cudaDeviceSynchronize());
                string filename = "part_4_link_f" + to_string(frame) + "_n" + to_string(cubesRes) + ".obj";
                WriteObjFile(frameSize, meshVertices_h + offset, meshNormals_h + offset, filename);
            }

            // testing
            if (correctTest)
            {
                checkCudaErrors(cudaDeviceSynchronize());
                TestCorrectness(frameSize, meshVertices_h + offset, meshNormals_h + offset, frame);
            }

            offset += frameSize;
            twist += 1.0 / float(frameNum) * maxTwist;
        }
        cudaDeviceSynchronize();
        end = high_resolution_clock::now();
        totalTime = (duration<double>(end - start)).count();

        printf("\nPart4\nTime taken: \nTotal: %f sec\n", totalTime);
        showMemUsage();
    }

    ///////////////////////////////////////////////////////////////////
    //                         Free memory                           //
    ///////////////////////////////////////////////////////////////////
    cudaFree(meshVertices_h);
    cudaFree(meshNormals_h);
}
