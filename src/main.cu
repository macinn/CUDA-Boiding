#include "BoidsDrawer.cu" 
#include "cuda/BoidsLogicGPU.cu"

int main()
{
	//BoidsLogicGPU logic(1000, 30, 30, 30);
	//logic.update(0.1f,0,0);

	const int WIDTH = 1920;
	const int HEIGHT = 1080;
	const int GL_VERSION_MAJOR = 4;
	const int GL_VERSION_MINOR = 4;

	uint N, size, type;
	std::cout << "Boids simulation, Skrzypczak Marcin" << std::endl;
	std::cout << "Choose simulation type: " << std::endl;
	std::cout << "1. CPU" << std::endl;
	std::cout << "2. GPU with grid" << std::endl;
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
