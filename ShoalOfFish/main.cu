#include "BoidsDrawer.cuh" 
#include "cuda/BoidsLogicGPU.cu"

static bool isCudaAvaialable()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	return deviceCount > 0;
}

int main()
{
	const int WIDTH = 1920;
	const int HEIGHT = 1080;
	const int GL_VERSION_MAJOR = 4;
	const int GL_VERSION_MINOR = 4;
	const bool cudaAvailable = isCudaAvaialable();

	uint N, size, type;
	std::cout << "Boids simulation, Skrzypczak Marcin" << std::endl;
	std::cout << "Choose simulation type: " << std::endl;
	std::cout << "1. CPU" << std::endl;
	if (cudaAvailable)
		std::cout << "2. GPU with grid" << std::endl;
	else 
		std::cout << "2. GPU with grid (no CUDA device found!)" << std::endl;
	std::cin >> type;
	std::cout << "Set container size: ";
	std::cin >> size;
	std::cout << "Set number of boids: ";
	std::cin >> N;

	// Create drawer class
	BoidsDrawer drawer("Shoal of fish",
		WIDTH, HEIGHT,
		GL_VERSION_MAJOR, GL_VERSION_MINOR,
		false, N, size);

	switch (type)
	{
	case(1):
	{
		drawer.setBoidsLogic(new BoidsLogic(N, size, size, size));
		break;
	}
	case(2):
	{
		drawer.setBoidsLogic(new BoidsLogicGPU(N, size, size, size));
		break;
	}
	default:
	{
		std::cout << "Wrong simulation type" << std::endl;
		return 0;
	}
	}

	// Main program	loop
	while (!drawer.getWindowShouldClose())
	{
		drawer.update();
		drawer.render();
	}

	return 0;
}

