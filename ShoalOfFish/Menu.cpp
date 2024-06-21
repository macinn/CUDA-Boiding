#include "BoidsDrawer.cu" 
#include "Menu.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <string>
#include <vector>

// private
bool Menu::isCudaAvaialable()
{
	int deviceCount;
	cudaError_t cerr = cudaGetDeviceCount(&deviceCount);
	return cerr == cudaSuccess && deviceCount > 0;
}
int Menu::printOptions(std::vector <MenuOption*> options, bool printUnavailble)
{
	int number = 0;
	for (int i = 0; i < options.size(); i++)
	{
		if (options[i]->isAvailable)
			std::cout << ++number << ". " << options[i]->name;
		else if (printUnavailble)
			std::cout << ++number << ". " << options[i]->name << " " << unavailblePolicyMessage;
		std::cout << std::endl;
	}
	return number;
}
MenuOption* Menu::selectOptions(std::string text, std::vector<MenuOption*> options, bool printUnavailble)
{
	std::cout << text << std::endl;
	if (printOptions(options, printUnavailble) == 0)
	{
		std::cout << "No available options!" << std::endl;
		return nullptr;
	}
	uint choice;
	bool isValid;
	do
	{
		std::cout << "> ";
		std::cin >> choice;
		isValid = !(choice < 1 || choice > options.size() || !options[choice - 1]->isAvailable);
		if (!isValid)
		{
			std::cout << "Invalid choice!" << std::endl;
		}
	} while (!isValid);
	return options[choice - 1];
}
void Menu::printDescription()
{
	std::cout << "\033[2J\033[H" << "Boids simulation, Skrzypczak Marcin" << std::endl;
	if (!cudaAvailable)
		std::cout << unavailblePolicyMessage << std::endl;
	std::cout << std::endl;

}
void Menu::createDrawer(uint N, uint size, BoidsLogic* logic)
{
	drawer = new BoidsDrawer("Shoal of fish",
		WIDTH, HEIGHT,
		GL_VERSION_MAJOR, GL_VERSION_MINOR,
		false, N, size);
	drawer->setBoidsLogic(logic);
}
void Menu::drawerLoop()
{
	while (!drawer->getWindowShouldClose())
	{
		drawer->update();
		drawer->render();
	}
	delete drawer;
}
void Menu::runDrawer()
{
	printDescription();
	LogicOption* policy = (LogicOption*)selectOptions("Choose execution policy:", availbleEngines, true);
	uint N, size;
	std::cout << "Enter number of boids: ";
	std::cin >> N;
	std::cout << "Enter container size: ";
	std::cin >> size;
	BoidsLogic* logic = policy->create(N, size, size, size);
	createDrawer(N, size, logic);
	drawerLoop();
}
void Menu::exit()
{
	running = false;
}
// public
Menu::~Menu()
{
	delete[] availbleEngines.data();
	delete[] availbleModes.data();
}
void Menu::run()
{
	while (running)
	{
		printDescription();
		ModeOption* mode = (ModeOption*)selectOptions("Choose action:", availbleModes, true);
		(this->*mode->run)();
	}
}

