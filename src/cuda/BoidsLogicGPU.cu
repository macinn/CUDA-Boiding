#include "BoidsLogic.cpp"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

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

__global__ void assignGridIndKernel(double gridSize, int gridSizeX, int gridSizeY, uint N, glm::vec3* dev_boids_p, int* dev_boids_grid_ind_1, int* dev_boids_grid_ind_2)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    while (idx < N)
    {
        int x = dev_boids_p[idx].x / gridSize;
        int y = dev_boids_p[idx].y / gridSize;
        int z = dev_boids_p[idx].z / gridSize;

        dev_boids_grid_ind_1[idx] = x + y * gridSizeX + z * gridSizeX * gridSizeY;
        dev_boids_grid_ind_2[idx] = dev_boids_grid_ind_1[idx];

        idx += BLOCK_SIZE * BLOCK_NUMBER;
    }

}

__device__ float distance2(glm::vec3 a, glm::vec3 b) {
    return (a.x - b.x) * (a.x - b.x)
		+ (a.y - b.y) * (a.y - b.y)
		+ (a.z - b.z) * (a.z - b.z);
}

__device__ float l2Norm(glm::vec3 a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

__global__ void updateBoidsKernel(const float dt, const uint N, 
    glm::vec3* dev_boids_p, glm::vec3* dev_boids_v, 
    const int* dev_boids_grid_ind, const int* dev_grid_start, const int* dev_grid_end,
    const int gridSizeX, const int gridSizeY,
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
        // TODO: more precise boundires
        for(int i_x = -1; i_x <= 1; i_x ++)
            for(int i_y = -1; i_y <= 1; i_y++)
                for (int i_z = -1; i_z <= 1; i_z++)
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

__global__ void findGridStartEnd(int* dev_grid_start, int* dev_grid_end, int* dev_boids_grid_ind_1, int gridCount)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < gridCount)
    {
        dev_grid_start[idx] = gridCount;
        dev_grid_end[idx] = -1;
    }

    // TODO: Check if regular find
    atomicMax(&dev_grid_end[idx], idx);
    atomicMin(&dev_grid_start[idx], idx);
}

class BoidsLogicGPU: public BoidsLogic {
private:
    glm::vec3* dev_boids_p;
    glm::vec3* dev_boids_v;
    cudaGraphicsResource* cuda_boids_p = NULL;
    cudaGraphicsResource* cuda_boids_v = NULL;
    int* dev_boids_grid_ind_1;
    int* dev_boids_grid_ind_2;
    int* dev_grid_start;
    int* dev_grid_end;
    double gridSize; 


    void init()
    {
  //      cudaError_t cudaStatus;
  //      curandGenerator_t gen;

  //      if (CURAND_STATUS_SUCCESS != curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT))
  //      {

  //      }

  //      if (CURAND_STATUS_SUCCESS != curandSetPseudoRandomGeneratorSeed(gen, time(NULL)))
  //      {

  //      }

  //      if (CURAND_STATUS_SUCCESS != curandGenerateUniform(gen, (float*)dev_boids_p, N * 3))
  //      {

		//}

        cudaError_t cudaStatus = cudaMemset((void**)&dev_boids_v, 0, N * sizeof(glm::vec3));
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed!");
        }
    }

    void updateData(float dt)
    {
        cudaError_t cudaStatus;

        int gridSizeX = (width - 1) / gridSize + 1;
        int gridSizeY = (height - 1) / gridSize + 1;

        updateBoidsKernel << < BLOCK_NUMBER, BLOCK_SIZE >> > (
             dt,  N,
             dev_boids_p,  dev_boids_v,
             dev_boids_grid_ind_1,  dev_grid_start,  dev_grid_end,
             gridSizeX,  gridSizeY,
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

        assignGridIndKernel << < BLOCK_NUMBER, BLOCK_SIZE >> > (this->gridSize, gridSizeX, gridSizeY, this->N, this->dev_boids_p,
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
        
        findGridStartEnd << < BLOCK_NUMBER, BLOCK_SIZE >> > (dev_grid_start, dev_grid_end, dev_boids_grid_ind_1, gridSizeX * gridSizeY * gridSizeZ);

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
	BoidsLogicGPU(uint N, uint width, uint height, uint depth = 0) :
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

        cudaStatus = cudaMemset((void**)&dev_boids_v, 0, N * sizeof(glm::vec3));
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed!");
        }

        cudaStatus = cudaMalloc((void**)&dev_boids_grid_ind_1, N * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed!");
        }

        cudaStatus = cudaMalloc((void**)&dev_boids_grid_ind_2, N * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed!");
        }

        cudaStatus = cudaMalloc((void**)&dev_grid_start, N * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed!");
        }

        cudaStatus = cudaMalloc((void**)&dev_grid_end, N * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed!");
        }
        gridSize = 2 * visualRange;
        // populate with random values
        this->init();
	}
    ~BoidsLogicGPU() {
        cudaFree(dev_boids_v);
        cudaFree(dev_boids_p);
    }

    // Update boids position and velocity
    void update(float dt, GLuint positionBuffer, GLuint velocityBuffer) {
        cudaError_t cudaStatus;

        if (cuda_boids_p == NULL)
        {
            cudaStatus = cudaGraphicsGLRegisterBuffer(&cuda_boids_p, positionBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
            if (cudaStatus != cudaSuccess) {
                throw std::runtime_error("cudaGraphicsGLRegisterBuffer failed!");
            }
        }
        if (cuda_boids_v == NULL)
        {
            cudaStatus = cudaGraphicsGLRegisterBuffer(&cuda_boids_v, velocityBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
            if (cudaStatus != cudaSuccess) {
                throw std::runtime_error("cudaGraphicsGLRegisterBuffer failed!");
            }
		}
        size_t size = N * sizeof(glm::vec3);
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

