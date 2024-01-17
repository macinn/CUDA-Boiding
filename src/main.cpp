#include "BoidsDrawer.cpp"

int main()
{
	const int WIDTH = 1920;
	const int HEIGHT = 1080;
	const int GL_VERSION_MAJOR = 4;
	const int GL_VERSION_MINOR = 4;

	uint N, size;
	std::cout << "Boids simulation, Skrzypczak Marcin" << std::endl;
	std::cout << "Set container size: ";
	std::cin >> size;
	std::cout << "Set number of boids: ";
	std::cin >> N;

	// Create drawer class
	BoidsDrawer drawer("Shoal of fish",
		WIDTH, HEIGHT,
		GL_VERSION_MAJOR, GL_VERSION_MINOR,
		false, N, size);

	// Main program loop
	while (!drawer.getWindowShouldClose())
	{
		drawer.update();
		drawer.render();
	}

	return 0;
}

