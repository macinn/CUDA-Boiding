#include <string>
#include <cuda_runtime_api.h>
#include <iostream>
#include <driver_types.h>
#include "BoidsDrawer.cu" 
#include "BoidsLogicGPU_SH.cuh"
#include "BoidsLogickCPU_P.cpp"
#include "BoidsLogicTEST.cpp"
#include "ProgressBar.cpp"
#include <vector>
#include <iomanip>
#include <numeric>
#include <chrono>

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
	LogicOption(std::string name, bool isAvailable, BoidsLogic* (*create)(unsigned, unsigned, unsigned, unsigned)): MenuOption(name, isAvailable)
	{
		this->create = create;
	}
	BoidsLogic* (*create)(unsigned, unsigned, unsigned, unsigned);
};
class ModeOption : public MenuOption
{
public:
	void (Menu::*run)();
	ModeOption(std::string name, void (Menu::*run)()) : MenuOption(name, true)
	{
		this->run = run;
	}
};

class Menu
{
	bool running = true;
	const bool cudaAvailable = isCudaAvaialable();
	std::vector<MenuOption*> availbleEngines = {
		new LogicOption("CPU", true, [](uint N, uint w, uint h, uint d) -> BoidsLogic*
			{return new BoidsLogic(N, w, h, d); }),
		new LogicOption("CPU parallel TEST", true, [](uint N, uint w, uint h, uint d) -> BoidsLogic*
			{return new BoidsLogicTEST(N, w, h, d); } ),
		new LogicOption("CPU parallel", true, [](uint N, uint w, uint h, uint d) -> BoidsLogic*
			{return new BoidsLogic(N, w, h, d); } ),
		new LogicOption("GPU with spatial hashing", cudaAvailable, [](uint N, uint w, uint h, uint d) -> BoidsLogic*
			{return new BoidsLogicCPU_P(N, w, h, d); })};
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

	static bool isCudaAvaialable()
	{
		int deviceCount;
		cudaError_t cerr = cudaGetDeviceCount(&deviceCount);
		return cerr == cudaSuccess && deviceCount > 0;
	}
	int printOptions(std::vector <MenuOption*> options, bool printUnavailble)
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
	MenuOption* selectOptions(std::string text, std::vector<MenuOption*> options, bool printUnavailble)
	{
		std::cout << text << std::endl;
		if(printOptions(options, printUnavailble) == 0)
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
		} while(!isValid);
		return options[choice - 1];
	}
	void printDescription() 
	{
		std::cout << "\033[2J\033[H" << "Boids simulation, Skrzypczak Marcin" << std::endl;
		if(!cudaAvailable)
			std::cout << unavailblePolicyMessage << std::endl;
		std::cout << std::endl ;
		
	}
	void createDrawer(uint N, uint size, BoidsLogic* logic)
	{
		drawer = new BoidsDrawer("Shoal of fish",
				WIDTH, HEIGHT,
				GL_VERSION_MAJOR, GL_VERSION_MINOR,
				false, N, size);
		drawer->setBoidsLogic(logic);
	}
	void drawerLoop()
	{
		while (!drawer->getWindowShouldClose())
		{
			drawer->update();
			drawer->render();
		}
		delete drawer;
	}
	void runBenchmark()
	{
		int numFrames = 100, numRuns = 3, N = 1000;
		double dt = 1 / 60;
		printDescription();
		std::cout << "Available engines:" << std::endl;
		int numOptions = printOptions(availbleEngines, false);
		std::cout << std::endl << "Each engine will start simulation from same starting conditions, calculate " << numFrames <<" frames, " << numRuns << " times." << std::endl;
		std::cout << "Starting conditions:" << std::endl << std::endl;
		std::cout << "Number of boids: " << N << std::endl;
		std::cout << "Container size: " << 30 << std::endl << std::endl;
		std::vector<BoidsLogic*> logics;
		std::vector<ProgressBar> pbs;
		std::vector<std::vector<long long>> results;
		for (int i = 0; i < numOptions; i++)
		{
			if (availbleEngines[i]->isAvailable)
			{
				logics.push_back(((LogicOption*)availbleEngines[i])->create(N, 30, 30, 30));
				pbs.push_back(ProgressBar(availbleEngines[i]->name, 70));
				results.push_back(std::vector<long long>());
			}
		}

		int numberSegments = numRuns * logics.size();
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
		GLuint instancePosition, instanceVelocity;
		if (!glfwInit()) {
			std::cerr << "Failed to initialize GLFW" << std::endl;
		}

		// Create a hidden off-screen OpenGL context with GLFW
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);  // Hide the window
		GLFWwindow* offscreenContext = glfwCreateWindow(1, 1, "Offscreen Context", nullptr, nullptr);
		if (!offscreenContext) {
			std::cerr << "Failed to create off-screen GLFW window" << std::endl;
			glfwTerminate();
		}

		// Make the off-screen context current
		glfwMakeContextCurrent(offscreenContext);
		GLenum glewError = glewInit();
		if (glewError != GLEW_OK) {
			std::cerr << "Failed to initialize GLEW: " << glewGetErrorString(glewError) << std::endl;
			glfwTerminate();
		}
		glGenBuffers(1, &instancePosition);
		glBindBuffer(GL_ARRAY_BUFFER, instancePosition);
		glBufferData(GL_ARRAY_BUFFER, N * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);
		glGenBuffers(1, &instanceVelocity);
		glBindBuffer(GL_ARRAY_BUFFER, instanceVelocity);
		glBufferData(GL_ARRAY_BUFFER, N * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);
		for (int i = 0; i < numOptions; i++)
		{
			pbs[i].display();
			//warmup
			for (int k = 0; k < numFrames/10; k++)
			{
				logics[i]->update(dt, instancePosition, instanceVelocity);
			}
			for (int j = 0; j < numRuns; j++)
			{
				auto start = std::chrono::high_resolution_clock::now();
				for (int k = 0; k < numFrames; k++)
				{
					logics[i]->update(dt, instancePosition, instanceVelocity);
				}
				auto end = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / numRuns;
				results[i].push_back(duration);
				pbs[i].incrementProgress(1. / numRuns);
				pbs[i].display();
			}
			std::cout << std::endl;
		};
		std::cout << std::endl << std::endl
			<< std::setw(40) << std::left << "name" 
			<< std::setw(10) << std::right << "mean" 
			<< std::setw(10) << std::right << "min" 
			<< std::setw(10) << std::right << "max" << std::endl;
		for (int i = 0; i < numOptions; i++)
		{
			double mean = std::accumulate(results[i].begin(), results[i].end(), 0.0) / results[i].size();
			double min = *std::min_element(results[i].begin(), results[i].end());
			double max = *std::max_element(results[i].begin(), results[i].end());

			std::cout << std::setw(40) << std::left << pbs[i].getLabel() 
				<< std::setw(10) << std::right << mean 
				<< std::setw(10) << std::right << min 
				<< std::setw(10) << std::right << max << std::endl;
		}
		std::cout << std::endl << std::endl << "Press any key to continue...";
		std::cin.clear();
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		std::cin.get();
	}
	void runDrawer()
	{
		printDescription();
		LogicOption* policy = (LogicOption*) selectOptions("Choose execution policy:", availbleEngines, true);
		uint N, size;
		std::cout << "Enter number of boids: ";
		std::cin >> N;
		std::cout << "Enter container size: ";
		std::cin >> size;
		BoidsLogic* logic = policy->create(N, size, size, size);
		createDrawer(N, size, logic);
		drawerLoop();
	}
	void exit()
	{
		running = false;
	}
public:
	~Menu()
	{
		delete[] availbleEngines.data();
		delete[] availbleModes.data();
	}
	void run()
	{
		while (running)
		{
			printDescription();
			ModeOption* mode = (ModeOption*)selectOptions("Choose action:", availbleModes, true);
			(this->*mode->run)();
		}
	}
};
