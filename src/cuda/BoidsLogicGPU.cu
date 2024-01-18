#include "BoidsLogic.cpp"
#include <cuda_runtime.h>
#include <random>

#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <stdexcept>

#define BLOCK_SIZE 1024
#define BLOCK_NUMBER 4

#pragma once

__device__ int clamp(int min, int x, int max)
{
    return x < min ? min : (x > max ? max : x);
}

__device__ float distance2(glm::vec3 a, glm::vec3 b) {
    return (a.x - b.x) * (a.x - b.x)
        + (a.y - b.y) * (a.y - b.y)
        + (a.z - b.z) * (a.z - b.z);
}

__device__ float l2Norm(glm::vec3 a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

__global__ void assignGridIndKernel(double gridSize, int gridSizeX, int gridSizeY, int gridSizeZ,
    uint N, glm::vec3* dev_boids_p, int* dev_boids_grid_ind_1, int* dev_boids_grid_ind_2)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    while (idx < N)
    {
        int x = dev_boids_p[idx].x / gridSize;
        int y = dev_boids_p[idx].y / gridSize;
        int z = dev_boids_p[idx].z / gridSize;

        x = clamp(0, x, gridSizeX - 1);
        y = clamp(0, y, gridSizeY - 1);
        z = clamp(0, z, gridSizeZ - 1);

        dev_boids_grid_ind_1[idx] = x + y * gridSizeX + z * gridSizeX * gridSizeY;
        dev_boids_grid_ind_2[idx] = dev_boids_grid_ind_1[idx];

        idx += BLOCK_SIZE * BLOCK_NUMBER;
    }
}

__global__ void updateBoidsKernel(const float dt, const uint N, 
    glm::vec3* dev_boids_p, glm::vec3* dev_boids_v, 
    const int* dev_boids_grid_ind, const int* dev_grid_start, const int* dev_grid_end,
    const int gridSizeX, const int gridSizeY, const int gridSizeZ,
    const float turnFactor, const float visualRange, const float protectedRange,
    const float centeringFactor, float avoidFactor, float matchingFactor,
    const float maxSpeed, const float minSpeed,
    const uint width, const uint height, const uint depth,
    const float marginFactor)
{
    float visualRangeSquared = visualRange * visualRange;
    float protectedRangeSquared = protectedRange * protectedRange;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    while (idx < N)
    {
        int current_grid_id = dev_boids_grid_ind[idx];
        uint countVisible = 0;
        uint countClose = 0;
        glm::vec3 vel = glm::vec3(0.0f);
        glm::vec3 center = glm::vec3(0.0f);
        glm::vec3 close = glm::vec3(0.0f);

        int ind_x = current_grid_id % gridSizeX;
        int ind_y = (current_grid_id / gridSizeX) % gridSizeY;
        int ind_z = current_grid_id / (gridSizeX * gridSizeY);

        for (int i_x = -(ind_x > 0); i_x <= (ind_x < gridSizeX); i_x++)
            for (int i_y = -(ind_y > 0); i_y <= (ind_y < gridSizeY); i_y++)
                for (int i_z = -(ind_z > 0); i_z <= (ind_z < gridSizeZ); i_z++)
                {
                    int neighbour_grid_id =
                        +i_x
                        + i_y * gridSizeX
                        + i_z * gridSizeX * gridSizeY;

                    for (int j = dev_grid_start[neighbour_grid_id]; j <= dev_grid_end[neighbour_grid_id]; j++)
                    {
                        if (idx != j) {
                            float distanceSquared = distance2(dev_boids_p[idx], dev_boids_p[j]);
                            if (distanceSquared < visualRangeSquared)
                            {
                                center += dev_boids_p[j];
                                countVisible++;

                                if (distanceSquared < protectedRangeSquared)
                                {
                                    vel += dev_boids_v[j];
                                    close -= dev_boids_p[j];
                                    countClose++;
                                }
                            }

                        }
                    }

                }

        if (countVisible > 0) {
            center /= countVisible;

            if (countClose > 0) {
                vel /= countClose;
            }
        }

        close += (float)countClose * dev_boids_p[idx];
        dev_boids_v[idx] +=
            (center - dev_boids_p[idx]) * centeringFactor	// cohesion
            + close * avoidFactor						// separation	
            + (vel - dev_boids_v[idx]) * matchingFactor;		// alignment

        if (dev_boids_p[idx].x < width * marginFactor) {
            dev_boids_v[idx].x += turnFactor;
        }
        if (dev_boids_p[idx].x > width * (1 - marginFactor)) {
            dev_boids_v[idx].x -= turnFactor;
        }
        if (dev_boids_p[idx].y < height * marginFactor) {
            dev_boids_v[idx].y += turnFactor;
        }
        if (dev_boids_p[idx].y > height * (1 - marginFactor)) {
            dev_boids_v[idx].y -= turnFactor;
        }
        if (dev_boids_p[idx].z < depth * marginFactor) {
            dev_boids_v[idx].z += turnFactor;
        }
        if (dev_boids_p[idx].z > depth * (1 - marginFactor)) {
            dev_boids_v[idx].z -= turnFactor;
        }
    
        float speed = l2Norm(dev_boids_v[idx]);
        if (speed > maxSpeed) {
            dev_boids_v[idx] /= speed;
            dev_boids_v[idx] *= maxSpeed;
        }
        else if (speed < minSpeed) {
            dev_boids_v[idx] /= speed;
            dev_boids_v[idx] *= minSpeed;
        }

        dev_boids_p[idx] += dev_boids_v[idx] * dt;

        idx += BLOCK_SIZE * BLOCK_NUMBER;
    }
}

__global__ void findGridStartEnd(int* dev_grid_start, int* dev_grid_end, int* dev_boids_grid_ind, int gridCount, uint N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < gridCount)
    {
        dev_grid_start[idx] = gridCount;
        dev_grid_end[idx] = -1;
    }
    while (idx < N)
    {
        if (idx == 0) {
            dev_grid_start[dev_boids_grid_ind[idx]] = 0;
        }
        else if (dev_boids_grid_ind[idx] != dev_boids_grid_ind[idx - 1])
        {
            dev_grid_end[dev_boids_grid_ind[idx - 1]] = idx - 1;
            dev_grid_start[dev_boids_grid_ind[idx]] = idx;

            if (idx == N - 1)
            {
                dev_grid_end[dev_boids_grid_ind[idx]] = idx;
            }
        }

        idx += BLOCK_SIZE * BLOCK_NUMBER;
    }
}

class BoidsLogicGPU: public BoidsLogic {
private:
    glm::vec3* dev_boids_p;
    glm::vec3* dev_boids_v;
    cudaGraphicsResource* cuda_boids_p;
    cudaGraphicsResource* cuda_boids_v;
    int* dev_boids_grid_ind_1;
    int* dev_boids_grid_ind_2;
    int* dev_grid_start;
    int* dev_grid_end;
    double gridSize;
    bool firstRun = true;


    void init()
    {
        cudaError_t cudaStatus;

        cudaStatus = cudaMemset(dev_boids_v, 0, N * sizeof(glm::vec3));
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed!");
        }

        glm::vec3* boids_p = new glm::vec3[N]();
        std::default_random_engine rd{ static_cast<long uint>(time(0)) };
        std::mt19937 gen{ rd() };
        std::uniform_real_distribution<> w(0, width);
        std::uniform_real_distribution<> h(0, height);
        std::uniform_real_distribution<> z(0, depth);

        for (uint i = 0; i < N; i++) {
            boids_p[i] = glm::vec3(w(gen), h(gen), z(gen));
        }

        cudaStatus = cudaMemcpy(dev_boids_v, boids_p, N * sizeof(glm::vec3), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy failed!");
        }

        delete[] boids_p;
    }

    void updateData(float dt)
    {
        cudaError_t cudaStatus;

        int gridSizeX = (width - 1) / gridSize + 1;
        int gridSizeY = (height - 1) / gridSize + 1;
        int gridSizeZ = (depth - 1) / gridSize + 1;

        updateBoidsKernel << < BLOCK_NUMBER, BLOCK_SIZE >> > (
             dt,  N,
             dev_boids_p,  dev_boids_v,
             dev_boids_grid_ind_1,  dev_grid_start,  dev_grid_end,
             gridSizeX,  gridSizeY, gridSizeZ,
             turnFactor,  visualRange,  protectedRange,
             centeringFactor,  avoidFactor,  matchingFactor,
             maxSpeed,  minSpeed,
             width,  height,  depth,
             marginFactor);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("updateBoidsKernel failed!");
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaDeviceSynchronize returned error after launching updateBoidsKernel!");
        }
    }

    void updateBuffers(GLuint positionBuffer, GLuint velocityBuffer)
    {

    }

    void assignGridInd()
    {
        cudaError_t cudaStatus;

        int gridSizeX = (width - 1) / gridSize + 1;
        int gridSizeY = (height - 1) / gridSize + 1;
        int gridSizeZ = (depth - 1) / gridSize + 1;

        assignGridIndKernel << < BLOCK_NUMBER, BLOCK_SIZE >> > (this->gridSize, gridSizeX, gridSizeY, gridSizeZ, this->N, this->dev_boids_p,
            dev_boids_grid_ind_1, dev_boids_grid_ind_2);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("assignGridIndKernel failed!");
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaDeviceSynchronize returned error after launching assignGridIndKernel!");
        }
    }

    void sortGrid()
    {
        thrust::sort_by_key(thrust::device, dev_boids_grid_ind_1, dev_boids_grid_ind_1 + N, dev_boids_v);
        thrust::sort_by_key(thrust::device, dev_boids_grid_ind_2, dev_boids_grid_ind_2 + N, dev_boids_p);

        cudaError_t cudaStatus;

        int gridSizeX = (width - 1) / gridSize + 1;
        int gridSizeY = (height - 1) / gridSize + 1;
        int gridSizeZ = (depth - 1) / gridSize + 1;
        
        findGridStartEnd << < BLOCK_NUMBER, BLOCK_SIZE >> > (dev_grid_start, dev_grid_end, dev_boids_grid_ind_1, gridSizeX * gridSizeY * gridSizeZ, N);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("findGridStartEnd failed!");
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaDeviceSynchronize returned error after launching findGridStartEnd!");
        }
    }

public:
	BoidsLogicGPU(uint N, uint width, uint height, uint depth) :
        BoidsLogic(N, width, height, depth)
	{
        cudaError_t cudaStatus;

        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaSetDevice failed!");
        }

        cudaStatus = cudaMalloc((void**)&dev_boids_v, N * sizeof(glm::vec3));
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed!");
        }

        cudaStatus = cudaMalloc((void**)&dev_boids_p, N * sizeof(glm::vec3));
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed!");
        }

        cudaStatus = cudaMalloc((void**)&dev_boids_grid_ind_1, N * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed!");
        }

        cudaStatus = cudaMalloc((void**)&dev_boids_grid_ind_2, N * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed!");
        }

        cudaStatus = cudaMalloc((void**)&dev_grid_start, N * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed!");
        }

        cudaStatus = cudaMalloc((void**)&dev_grid_end, N * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed!");
        }
        gridSize = 2 * visualRange;



        
        // populate with random values
        this->init();
	}
    ~BoidsLogicGPU() {
        cudaFree(dev_boids_v);
        cudaFree(dev_boids_p);
        cudaFree(dev_boids_grid_ind_1);
        cudaFree(dev_boids_grid_ind_2);
        cudaFree(dev_grid_start);
        cudaFree(dev_grid_end);
    }

    // Update boids position and velocity
    void update(float dt, GLuint positionBuffer, GLuint velocityBuffer) {
        cudaError_t cudaStatus;
        if (firstRun)
        {
            firstRun = false;
            cudaStatus = cudaGraphicsGLRegisterBuffer(&cuda_boids_p, positionBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
            if (cudaStatus != cudaSuccess) {
                throw std::runtime_error("cudaGraphicsGLRegisterBuffer failed!");
            }


            cudaStatus = cudaGraphicsGLRegisterBuffer(&cuda_boids_v, velocityBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
            if (cudaStatus != cudaSuccess) {
                throw std::runtime_error("cudaGraphicsGLRegisterBuffer failed!");
            }

        }
        size_t size;
        assignGridInd();
        sortGrid();

        cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&dev_boids_p, &size, cuda_boids_p);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaGraphicsResourceGetMappedPointer failed!");
        }
        cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&dev_boids_v, &size, cuda_boids_v);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaGraphicsResourceGetMappedPointer failed!");
        }

        updateData(dt);

        cudaStatus = cudaGraphicsUnmapResources(1, &cuda_boids_p, 0);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaGraphicsUnmapResources failed!");
        }
        cudaStatus = cudaGraphicsUnmapResources(1, &cuda_boids_v, 0);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaGraphicsUnmapResources failed!");
        }
        //updateBuffers(positionBuffer, velocityBuffer);
    }
};

