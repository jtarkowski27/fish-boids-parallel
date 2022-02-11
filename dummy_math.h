#pragma once
#include <helper_math.h>

__device__ __host__ float3 zero3()
{
    return make_float3(0.0f);
}

__device__ __host__ float3 limit3(const float3 a, const float max)
{
    float length = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
    if (length > max) {
        float newLength = max / length;
        return a * newLength;
    }
    return a;
}

float range(float min, float max)
{
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = max - min;
    float r = random * diff;
    return min + r;
}

__host__ __device__ float3 rotate(float3 &v, float3 &k, double theta)
{
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);

    float3 rotated = (v * cos_theta) + (cross(k, v) * sin_theta) + (k * dot(k, v)) * (1 - cos_theta);

    return rotated;
}