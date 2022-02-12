#pragma once

#include "boids.cuh"
#include "dummy_math.cuh"

struct steer_args
{
    float3 flock_direction;
    float3 mass_centre;
    float3 avoidance_direction;

    float3 position;
    int floackmates_count;
    float separation;
    float alignment;

    float cohesion;
};

#define FACTOR .05;

__host__ __device__ void
assign_vertex(float *vec3, float3 v, int k)
{
    vec3[k + 0] = v.x;
    vec3[k + 1] = v.y;
    vec3[k + 2] = v.z;
}

__host__ __device__ void calculate_triangle_indices(boids_s &boids, int i)
{
    int k = i * INDICES_PER_TRIANGLE * TRIANGLES_PER_FISH * DIMENSIONS_COUNT;

    float3 up = make_float3(0, 1, 0);

    float3 p = get_float3(boids.positions, i);
    float3 dir = get_float3(boids.directions, i);

    float3 normal = normalize(cross(dir, up));

    float3 vb = rotate(dir, normal, 1.1);
    float3 vc = rotate(vb, dir, 2 * M_PI / 3);
    float3 vd = rotate(vc, dir, 2 * M_PI / 3);

    float size = boids.controls.fish_size * (boids.types[i] == FishType::Predator ? 2 : 1);

    float3 a = p + dir * 3 * size;
    float3 b = p + vb * size;
    float3 c = p + vc * size;
    float3 d = p + vd * size;

    assign_vertex(boids.geometry, a, k + 0 + 9 * 0);
    assign_vertex(boids.geometry, b, k + 3 + 9 * 0);
    assign_vertex(boids.geometry, c, k + 6 + 9 * 0);

    assign_vertex(boids.geometry, a, k + 0 + 9 * 1);
    assign_vertex(boids.geometry, c, k + 3 + 9 * 1);
    assign_vertex(boids.geometry, d, k + 6 + 9 * 1);

    assign_vertex(boids.geometry, a, k + 0 + 9 * 2);
    assign_vertex(boids.geometry, d, k + 3 + 9 * 2);
    assign_vertex(boids.geometry, b, k + 6 + 9 * 2);

    assign_vertex(boids.geometry, b, k + 0 + 9 * 3);
    assign_vertex(boids.geometry, c, k + 3 + 9 * 3);
    assign_vertex(boids.geometry, d, k + 6 + 9 * 3);
}

__device__ __host__ float3 check_boundaries(float3 p)
{
    float3 acceleration = zero3();

    const float get_back = 2.0f;
    const float min = -1.0f;
    const float max = -min;

    float dist = dot(p, p);

    float acc_v = get_back * dist;

    if (p.x < min)
    {
        acceleration.x = acc_v;
    }
    else if (p.x > max)
    {
        acceleration.x = -acc_v;
    }

    if (p.y < min)
    {
        acceleration.y = acc_v;
    }
    else if (p.y > max)
    {
        acceleration.y = -acc_v;
    }

    if (p.z < min)
    {
        acceleration.z = acc_v;
    }
    else if (p.z > max)
    {
        acceleration.z = -acc_v;
    }

    return acceleration;
}

__device__ __host__ float3 calculate_force(steer_args args)
{
    args.flock_direction /= args.floackmates_count;

    args.mass_centre /= args.floackmates_count;
    args.mass_centre -= args.position;

    float3 separation_f3 = limit3(args.avoidance_direction, 1.f) * args.separation;
    float3 alignment_f3 = limit3(args.flock_direction, 1.f) * args.alignment;
    float3 cohesion_f3 = limit3(args.mass_centre, 1.f) * args.cohesion;

    float3 force = separation_f3 + alignment_f3 + cohesion_f3;

    return force;
}

__device__ __host__ float3 perp(float3 v)
{
    float min = abs(v.x);
    float3 cardinal_direction = make_float3(1.0f, 0.0f, 0.0f);

    if (abs(v.y) < min)
    {
        min = abs(v.y);
        cardinal_direction = make_float3(0.0f, 1.0f, 0.0f);
    }

    if (abs(v.z) < min)
    {
        cardinal_direction = make_float3(0.0f, 0.0f, 1.0f);
    }

    return cross(v, cardinal_direction);
}

// Steer one fish based on all other fishes
__device__ __host__ void steer_prey(boids_s boids, float dt, int id)
{
    const float viewRadius = boids.types[id] == FishType::Predator ? boids.controls.predator_view_radius : 0.1f;
    const float avoidRadius = boids.types[id] == FishType::Predator ? 0.1f : 0.05f;

    const float maxSpeed = 1.0f;

    float3 flock_direction = zero3();
    float3 avoidance_direction = zero3();
    float3 mass_centre = zero3();

    int floackmates_count = 0;

    float3 position = get_float3(boids.positions, id);
    float3 velocity = get_float3(boids.velocities, id);
    float3 direction = get_float3(boids.directions, id);

    float weight = boids.weights[id];

    for (int i = boids.predators_count; i < boids.n; i++)
    {
        if (i == id)
        {
            continue;
        }

        float3 fish_pos = get_float3(boids.positions, i);
        float3 offset = fish_pos - position;

        float distance_squared = dot(offset, offset);

        if (distance_squared > viewRadius * viewRadius)
        {
            continue;
        }

        floackmates_count++;

        float3 fish_direction = get_float3(boids.directions, i);
        flock_direction += fish_direction;
        mass_centre += fish_pos;

        if (distance_squared < avoidRadius * avoidRadius)
        {
            avoidance_direction -= offset;
        }
    }

    if (floackmates_count > 0)
    {
        steer_args args{
            flock_direction,
            mass_centre,
            avoidance_direction,
            position,
            floackmates_count,
            boids.controls.separation,
            boids.controls.alignment,
            boids.controls.cohesion,
        };

        float3 force = calculate_force(args);
        float3 accelerationeleration = force * weight;

        velocity += accelerationeleration * dt;
        velocity = limit3(velocity, maxSpeed);
    }

    float3 bound_force = check_boundaries(position);
    velocity += bound_force / weight * dt;

    // Predator avoidance
    int close_predators_count = 0;
    float3 avoid_danger = zero3();
    float predator_min_dist = boids.types[id] == FishType::Predator ? .05 : boids.controls.predator_avoidance_radius;

    for (int i = 0; i < boids.predators_count; i++)
    {
        if (i == id)
        {
            continue;
        }

        float3 predator_position = get_float3(boids.positions, i);
        float3 avoid_vector = position - predator_position;

        float dist = dot(avoid_vector, avoid_vector);

        if (sqrt(dist) > predator_min_dist)
        {
            continue;
        }

        close_predators_count++;

        avoid_danger += avoid_vector / dist;
    }

    if (close_predators_count > 0)
    {
        float avoidance_factor = boids.types[id] == FishType::Predator ? 1.0f : boids.controls.predator_avoidance;

        float3 accelerationeleration = avoid_danger * weight * avoidance_factor;
        velocity += accelerationeleration * dt;
        velocity = limit3(velocity, maxSpeed);
    }

    bound_force = check_boundaries(position);
    velocity += bound_force / weight * dt;

    add_float3(boids.positions, id, velocity * dt);
    assign_float3(boids.velocities, id, velocity);
    assign_float3(boids.directions, id, normalize(velocity));
}

// Main kernel, considers each fish parallelly
__global__ void boids_step_gpu(boids_s boids, float dt, bool debug = false)
{
    int threads_per_block = blockDim.x * blockDim.y;
    int block_idx_in_grid = blockIdx.x + gridDim.x * blockIdx.y;
    int thread_idx_in_block = threadIdx.x + blockDim.x * threadIdx.y;
    int tid = block_idx_in_grid * threads_per_block + thread_idx_in_block;

    if (tid < boids.n)
    {
        boids.types[tid] = tid < boids.predators_count ? FishType::Predator : FishType::Prey;

        if (boids.controls.is_running)
        {
            steer_prey(boids, boids.controls.animation_speed * dt, tid);
        }

        calculate_triangle_indices(boids, tid);
    }
}

__device__ __host__ void boids_step_cpu(boids_s boids, float dt, bool debug = false)
{
    for (int i = 0; i < boids.n; i++)
    {
        boids.types[i] = i < boids.predators_count ? FishType::Predator : FishType::Prey;

        if (boids.controls.is_running)
        {
            steer_prey(boids, boids.controls.animation_speed * dt, i);
        }

        calculate_triangle_indices(boids, i);
    }
}