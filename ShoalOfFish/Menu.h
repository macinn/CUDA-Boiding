#pragma once

#include <string>
#include "BoidsDrawer.cu" 
#include "BoidsEngine_GPU_SH.cuh"
#include "BoidsEngine_CPU_P.cpp"
#include "BoidsEngine_CPU_TEST.cpp"
#include <vector>

class Menu;

using uint = unsigned int;

class MenuOption
{
public:
	std::string name;
	bool isAvailable;
	MenuOption(std::string name, bool isAvailable)
	{
		this->name = name;
		this->isAvailable = isAvailable;
	}
};

class LogicOption : public MenuOption
{
public:
	LogicOption(std::string name, bool isAvailable, BoidsEngine_CPU* (*create)(unsigned, unsigned, unsigned, unsigned)) : MenuOption(name, isAvailable)
	{
		this->create = create;
	}
	BoidsEngine_CPU* (*create)(unsigned, unsigned, unsigned, unsigned);
};

class ModeOption : public MenuOption
{
public:
	void (Menu::* run)();
	ModeOption(std::string name, void (Menu::* run)()) : MenuOption(name, true)
	{
		this->run = run;
	}
};

class Menu
{
	bool running = true;
	const bool cudaAvailable = isCudaAvaialable();
	std::vector<MenuOption*> availbleEngines = {
		new LogicOption("CPU", true, [](uint N, uint w, uint h, uint d) -> BoidsEngine_CPU*
			{return new BoidsEngine_CPU(N, w, h, d); }),
		new LogicOption("CPU parallel TEST", true, [](uint N, uint w, uint h, uint d) -> BoidsEngine_CPU*
			{return new BoidsLogicTEST(N, w, h, d); }),
		new LogicOption("CPU parallel", true, [](uint N, uint w, uint h, uint d) -> BoidsEngine_CPU*
			{return new BoidsLogicCPU_P(N, w, h, d); }),
		new LogicOption("GPU with spatial hashing", cudaAvailable, [](uint N, uint w, uint h, uint d) -> BoidsEngine_CPU*
			{return new BoidsLogicGPU_SH(N, w, h, d); }) };
	std::vector<MenuOption*> availbleModes = {
		new ModeOption{"Run simulation", &Menu::runDrawer},
		new ModeOption{"Run benchmark", &Menu::runBenchmark},
		new ModeOption{"Exit", &Menu::exit} };
	const std::string unavailblePolicyMessage = "[No CUDA devide found!]";

	const int WIDTH = 1920;
	const int HEIGHT = 1080;
	const int GL_VERSION_MAJOR = 4;
	const int GL_VERSION_MINOR = 4;
	BoidsDrawer* drawer;

	static bool isCudaAvaialable();
	int printOptions(std::vector <MenuOption*> options, bool printUnavailble);
	MenuOption* selectOptions(std::string text, std::vector<MenuOption*> options, bool printUnavailble);
	void printDescription();
	void createDrawer(uint N, uint size, BoidsEngine_CPU* logic);
	void drawerLoop();
	void runBenchmark();
	void runDrawer();
	void exit();
public:
	~Menu();
	void run();
};
