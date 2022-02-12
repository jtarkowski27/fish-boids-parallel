#pragma once

#include <vector_types.h>
#include <helper_math.h>
#include <helper_timer.h>
#include <helper_cuda.h>

struct vec3_s
{
    int n = 0;

    float *x;
    float *y;
    float *z;
};

__host__ void free_host_device(void *ptr, cudaMemoryType type)
{
    switch (type)
    {
    case cudaMemoryType::cudaMemoryTypeHost:
        printf("free cudaMemoryTypeHost\n");
        free(ptr);
        break;
    case cudaMemoryType::cudaMemoryTypeDevice:
        printf("cudaFree cudaMemoryTypeDevice\n");
        cudaFree(ptr);
        break;

    default:
        break;
    }
}

__host__ void malloc_data(vec3_s &vec3, int n, cudaMemoryType type)
{
    vec3.n = n;

    switch (type)
    {
    case cudaMemoryType::cudaMemoryTypeHost:
        vec3.x = (float *)malloc(n * sizeof(float));
        vec3.y = (float *)malloc(n * sizeof(float));
        vec3.z = (float *)malloc(n * sizeof(float));
        break;
    case cudaMemoryType::cudaMemoryTypeDevice:
        checkCudaErrors(cudaMalloc((void **)&vec3.x, n * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&vec3.y, n * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&vec3.z, n * sizeof(float)));
        break;
    default:
        break;
    }
}

__host__ void free_data(vec3_s &vec3, cudaMemoryType type)
{
    switch (type)
    {
    case cudaMemoryType::cudaMemoryTypeHost:
        free(vec3.x);
        free(vec3.y);
        free(vec3.z);
        break;
    case cudaMemoryType::cudaMemoryTypeDevice:
        cudaFree(vec3.x);
        cudaFree(vec3.y);
        cudaFree(vec3.z);
        break;
    default:
        break;
    }
}

__host__ __device__ float3 get_float3(vec3_s &vec3, int i)
{
    float x = vec3.x[i];
    float y = vec3.y[i];
    float z = vec3.z[i];

    return make_float3(x, y, z);
}

__host__ __device__ void assign_float3(vec3_s &vec3, int i, float3 v)
{
    vec3.x[i] = v.x;
    vec3.y[i] = v.y;
    vec3.z[i] = v.z;
}

__host__ __device__ void add_float3(vec3_s &vec3, int i, float3 v)
{
    vec3.x[i] += v.x;
    vec3.y[i] += v.y;
    vec3.z[i] += v.z;
}

__host__ void memcpy_device_data(vec3_s &target, vec3_s &source, cudaMemcpyKind kind)
{
    target.n = source.n;

    checkCudaErrors(cudaMemcpy(target.x, source.x, sizeof(float) * source.n, kind));
    checkCudaErrors(cudaMemcpy(target.y, source.y, sizeof(float) * source.n, kind));
    checkCudaErrors(cudaMemcpy(target.z, source.z, sizeof(float) * source.n, kind));
}