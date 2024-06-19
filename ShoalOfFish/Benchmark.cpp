#include <chrono>
#include <iomanip>
#include <numeric>
#include <iostream>
#include <vector>

#include "Menu.h"
#include "ProgressBar.cpp"
#include "BoidsEngine.cpp"

void Menu::runBenchmark()
{
	int numFrames = 100, numRuns = 3, N = 1000;
	double dt = 1 / 60;
	printDescription();
	std::cout << "Available engines:" << std::endl;
	int numOptions = printOptions(availbleEngines, false);
	std::cout << std::endl << "Each engine will start simulation from same starting conditions, calculate " << numFrames << " frames, " << numRuns << " times." << std::endl;
	std::cout << "Starting conditions:" << std::endl << std::endl;
	std::cout << "Number of boids: " << N << std::endl;
	std::cout << "Container size: " << 30 << std::endl << std::endl;
	std::vector<BoidsEngine_CPU*> logics;
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
		for (int k = 0; k < numFrames / 10; k++)
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