#pragma once
#include <string>
using namespace std;

void WriteObjFile(
    int frameSize,
    float3 *meshVertices,
    float3 *meshNormals,
    string const& fileName)
{
	// FILE faster than streams.

	FILE* file = fopen(fileName.c_str(), "w");
	if (!file)
	{
		throw runtime_error("Could not write obj file.");
	}

	// write stats
	// fprintf(file, "# %d vertices, %d triangles\n\n",
	// 	static_cast<int>(frameSize),
	// 	static_cast<int>(frameSize));

    int nvertices = 0;

	// vertices
	for (int vi = 0; vi < frameSize; ++vi)
	{
		float3 const& v = meshVertices[vi];
        if (v.x != 0.0 || v.y != 0.0 || v.z != 0.0) {
		    fprintf(file, "v %f %f %f\n", v.x, v.y, v.z);
            nvertices++;
        }
	}

	// vertex normals
	fprintf(file, "\n");
	for (int ni = 0; ni < frameSize; ++ni)
	{
		float3 const& vn = meshNormals[ni];
        if (vn.x != 0.0 || vn.y != 0.0 || vn.z != 0.0)
		    fprintf(file, "vn %f %f %f\n", vn.x, vn.y, vn.z);
	}

	// triangles (1-based)
	fprintf(file, "\n");
	for (int ti = 0; ti < nvertices; ti += 3)
	{
		// int3 const& t = meshTriangles[ti];
		fprintf(file, "f %d//%d %d//%d %d//%d\n",
			ti + 1, ti + 1,
			ti + 2, ti + 2,
			ti + 3, ti + 3);
	}

	fclose(file);
}