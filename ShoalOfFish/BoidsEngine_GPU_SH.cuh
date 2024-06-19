#pragma once

#pragma once

#include "BoidsEngine.cpp"
#include <random>

#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>

#include <stdexcept>
#include <thrust/execution_policy.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <ostream>
#include <crt/host_defines.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/detail/sort.inl>
#include <thrust/system/cuda/detail/guarded_driver_types.h>
#include <GL/glew.h>
#include <glm/ext/vector_float3.hpp>

#define BLOCK_SIZE 1024
#define BLOCK_NUMBER 12

class BoidsLogicGPU_SH : public BoidsEngine_CPU {
private:
    glm::vec3* dev_boids_p;
    glm::vec3* dev_boids_v;
    cudaGraphicsResource* cuda_boids_p = NULL;
    cudaGraphicsResource* cuda_boids_v = NULL;
    // two index arrays for sorting velocity and position
    int* dev_boids_grid_ind_1;
    int* dev_boids_grid_ind_2;
    int* dev_grid_start;
    int* dev_grid_end;
    double gridSize;
    bool firstRun = true;

    int gridSizeX;
    int gridSizeY;
    int gridSizeZ;

    // initialize boids position and velocity
    void init();

    // update boids position and velocity
    void updateData(float dt);

    // calculate grid index of boids
    void assignGridInd();

    // sort boids by grid index and calculate start and end index of each grid
    void sortGrid();

public:
    // Constructor and destructor
    BoidsLogicGPU_SH(unsigned int N, unsigned int width, unsigned int height, unsigned int depth);
    ~BoidsLogicGPU_SH();

    // Update boids position and velocity
    void update(float dt, GLuint positionBuffer, GLuint velocityBuffer) override;

    // Set visual range, update grid parameters
    void setVisualRange(float visualRange) override;
};

