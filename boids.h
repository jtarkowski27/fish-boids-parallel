#pragma once

#include <vector_types.h>
#include <helper_math.h>
#include <helper_timer.h>
#include <helper_cuda.h>

#include "vec3.h"
#include "dummy_math.h"

#define TRIANGLES_PER_FISH 4
#define DIMENSIONS_COUNT 3
#define INDICES_PER_TRIANGLE 3

#define BOIDS_N 20000

enum FishType
{
    Prey,
    Predator,
};

struct simulation_controls_s
{
    bool is_running = true;

    float separation = 19.2f;
    float alignment = 11.7f;
    float cohesion = 1.3f;

    float predator_avoidance = 10.0f;
    float predator_view_radius = 0.5f;
    float predator_avoidance_radius = 0.3f;

    float fish_size = 0.018f;

    float animation_speed = 1.0f;
};

struct boids_s
{
    int n;
    int predators_count = 10;
    int indices_n;

    simulation_controls_s controls;

    FishType *types;

    vec3_s positions;
    vec3_s velocities;
    vec3_s directions;

    float *weights;
    float *geometry;
};

__host__ void free_data(boids_s &boids, cudaMemoryType type)
{
    free_data(boids.directions, type);
    free_data(boids.velocities, type);
    free_data(boids.positions, type);

    free_host_device(boids.geometry, type);
    free_host_device(boids.weights, type);
    free_host_device(boids.types, type);
}

__host__ void malloc_data(boids_s &boids, int n, cudaMemoryType type)
{
    boids.n = n;

    malloc_data(boids.directions, n, type);
    malloc_data(boids.velocities, n, type);
    malloc_data(boids.positions, n, type);

    boids.indices_n = n * TRIANGLES_PER_FISH * INDICES_PER_TRIANGLE * DIMENSIONS_COUNT;

    switch (type)
    {
    case cudaMemoryType::cudaMemoryTypeHost:
        boids.geometry = (float *)malloc(boids.indices_n * sizeof(float));
        boids.weights = (float *)malloc(boids.n * sizeof(float));
        boids.types = (FishType *)malloc(boids.n * sizeof(FishType));
        break;
    case cudaMemoryType::cudaMemoryTypeDevice:
        checkCudaErrors(cudaMalloc((void **)&boids.geometry, boids.indices_n * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&boids.weights, boids.n * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&boids.types, boids.n * sizeof(FishType)));
        break;
    default:
        break;
    }
}

__host__ void memcpy_data(boids_s &target, boids_s &source, cudaMemcpyKind kind)
{
    target.n = source.n;
    target.indices_n = source.indices_n;
    target.controls = source.controls;
    target.predators_count = source.predators_count;

    checkCudaErrors(cudaMemcpy(target.geometry, source.geometry, sizeof(float) * source.indices_n, kind));
    checkCudaErrors(cudaMemcpy(target.weights, source.weights, sizeof(float) * source.n, kind));
    checkCudaErrors(cudaMemcpy(target.types, source.types, sizeof(FishType) * source.n, kind));

    memcpy_device_data(target.directions, source.directions, kind);
    memcpy_device_data(target.velocities, source.velocities, kind);
    memcpy_device_data(target.positions, source.positions, kind);
}

void randomize_data(boids_s &boids, int n, int predators_count = 0)
{
    const float max_velocity = 0.2f;
    const float min_mass = 0.5f;
    const float max_mass = 5.0f;

    for (int i = 0; i < n; i++)
    {
        boids.positions.x[i] = range(-1.0f, 1.0f);
        boids.positions.y[i] = range(-1.0f, 1.0f);
        boids.positions.z[i] = range(-1.0f, 1.0f);

        boids.velocities.x[i] = range(-max_velocity, max_velocity);
        boids.velocities.y[i] = range(-max_velocity, max_velocity);
        boids.velocities.z[i] = range(-max_velocity, max_velocity);

        boids.weights[i] = range(min_mass, max_mass);

        float3 direction = make_float3(boids.velocities.x[i], boids.velocities.y[i], boids.velocities.z[i]);
        direction = normalize(direction);

        boids.directions.x[i] = direction.x;
        boids.directions.y[i] = direction.y;
        boids.directions.z[i] = direction.z;
    }

    for (int i = 0; i < n; i++)
    {
        boids.types[i] = i < predators_count ? FishType::Predator : FishType::Prey;
        boids.types[i] = i < predators_count ? FishType::Prey : FishType::Predator;
    }
}

void randomize_data(boids_s &boids)
{
    randomize_data(boids, boids.n);
}