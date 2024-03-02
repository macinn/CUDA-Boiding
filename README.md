# Shoal-of-fish
## Abstracct
Visualize 3D simulation of a large shoal of fish.

## Description
CPU and GPU-based boiding simulation, rendered using OpenGL.
### Boids algorithm
As with most artificial life simulations, Boids is an example of emergent behavior; that is, the complexity of Boids arises from the interaction of individual agents adhering to a set of simple rules. The rules applied in the simplest Boids world are as follows:

-**separation:** steer to avoid crowding local flockmates

-**alignment:** steer towards the average heading of local flockmates

-**cohesion:** steer to move towards the average position (center of mass) of local flockmates

## GPU Implementation
Spatial hashing was implemented to omit a major bottleneck-random memory access looking for nearby flockmates. Start/end indicies array were used to allow constant time access to given cell. Entire informatation about boids, i.e. position and valocity is stored on the GPU side. Data to OpenGl is directly transfered using [CUDA buffers](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html). 
Boids are simulated in a one CUDA thread - one boid manner.


## Examples
![image](https://github.com/macinn/CUDA-Boiding/assets/118574079/a9de632f-a79c-4e2f-90d2-8714c8b34516)

![image](https://github.com/macinn/CUDA-Boiding/assets/118574079/2b2637e8-3df7-4938-bb2f-a33bf4a9bdbb)

![image](https://github.com/macinn/CUDA-Boiding/assets/118574079/79a557a5-020e-42ab-83eb-c0aa562bc488)


