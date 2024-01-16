#include"libs.h"
#include "Shader.h"
#include "Model.cpp"
#include "Flock.cpp"
#include "Camera.cpp"
#include "Box.cpp"
#include "backends/imgui_impl_opengl3.h"
#include "backends/imgui_impl_glfw.h"
#pragma once

// TODO: destructor
class BoidDrawer
{
private:
	GLFWwindow* window;
	const int WINDOW_WIDTH;
	const int WINDOW_HEIGHT;
	int framebufferWidth;
	int framebufferHeight;
	Shader* shader;
	Shader* boxShader;

	//OpenGL Context
	const int GL_VERSION_MAJOR;
	const int GL_VERSION_MINOR;

	//Camera
	Camera* camera;

	//Matrices
	glm::mat4 ViewMatrix;
	glm::vec3 camPosition;
	glm::vec3 worldUp;
	glm::vec3 camFront;

	glm::mat4 ProjectionMatrix;
	float fov;
	float nearPlane;
	float farPlane;

	//Model
	InstancedPyramid* model;
	Flock* flock;
	Box* box;

	float dt = 0.f;
	float curTime = glfwGetTime();
	float lastTime = curTime;
	
	//Mouse Input
	double lastMouseX;
	double lastMouseY;
	double mouseX;
	double mouseY;
	double mouseOffsetX;
	double mouseOffsetY;
	bool firstMouse;
	bool spacePressed = false;

public:
	BoidDrawer() = default;
	~BoidDrawer()
	{
		//glfwDestroyWindow(this->window);
		//glfwTerminate();

		delete this->shader;
		delete this->boxShader;
		delete this->camera;
		delete this->model;
		delete this->flock;
		delete this->box;
	}

	static void framebuffer_resize_callback(GLFWwindow* window, int fbW, int fbH)
	{
		glViewport(0, 0, fbW, fbH);
	}

	void initWindow(const char* title, bool resizable)
	{
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, this->GL_VERSION_MAJOR);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, this->GL_VERSION_MINOR);
		glfwWindowHint(GLFW_RESIZABLE, resizable);


		this->window = glfwCreateWindow(this->WINDOW_WIDTH, this->WINDOW_HEIGHT, title, NULL, NULL);

		if (this->window == nullptr)
		{
			std::cout << "ERROR::GLFW_WINDOW_INIT_FAILED" << "\n";
			glfwTerminate();
		}

		glfwGetFramebufferSize(this->window, &this->framebufferWidth, &this->framebufferHeight);
		glfwSetFramebufferSizeCallback(window, BoidDrawer::framebuffer_resize_callback);

		glfwMakeContextCurrent(this->window);
	}

	void initGLFW()
	{
		if (glfwInit() == GLFW_FALSE)
		{
			std::cout << "ERROR::GLFW_INIT_FAILED" << "\n";
			glfwTerminate();
		}
	}

	void initGLEW()
	{
		glewExperimental = GL_TRUE;

		if (glewInit() != GLEW_OK)
		{
			std::cout << "ERROR::MAIN.CPP::GLEW_INIT_FAILED" << "\n";
			glfwTerminate();
		}
	}

	void initOpenGLOptions()
	{
		glEnable(GL_DEPTH_TEST);

		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
		glFrontFace(GL_CCW);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}

	void initMatrices()
	{
		this->ViewMatrix = glm::mat4(1.f);
		this->ViewMatrix = glm::lookAt(this->camPosition, this->camPosition + this->camFront, this->worldUp);

		this->ProjectionMatrix = glm::mat4(1.f);
		this->ProjectionMatrix = glm::perspective(
			glm::radians(this->fov),
			static_cast<float>(this->framebufferWidth) / this->framebufferHeight,
			this->nearPlane,
			this->farPlane
		);
	}

	void initShaders()
	{
		// TODO: change version inside of shader
		this->shader = new Shader("vertex_core.glsl", "fragment_core.glsl");
		this->boxShader = new Shader("box/vertex_box.glsl", "box/fragment_box.glsl");
	}

	void initLight()
	{
		this->shader->setVec3f(this->camPosition, "lightPos0");
	}

	void initUniforms()
	{
		//INIT UNIFORMS
		this->shader->setMat4fv(ViewMatrix, "ViewMatrix");
		this->shader->setMat4fv(ProjectionMatrix, "ProjectionMatrix");
		glm::mat4 ModelMatrix(1.f);
		this->shader->setMat4fv(ModelMatrix, "ModelMatrix");
		this->initLight();
	}

	BoidDrawer(const char* title,
		const int WINDOW_WIDTH, const int WINDOW_HEIGHT,
		const int GL_VERSION_MAJOR, const int GL_VERSION_MINOR,
		bool resizable
		) :WINDOW_WIDTH(WINDOW_WIDTH),
		WINDOW_HEIGHT(WINDOW_HEIGHT),
		GL_VERSION_MAJOR(GL_VERSION_MAJOR),
		GL_VERSION_MINOR(GL_VERSION_MINOR)
	{
		//Init variables
		this->window = nullptr;
		this->framebufferWidth = this->WINDOW_WIDTH;
		this->framebufferHeight = this->WINDOW_HEIGHT;

		this->camPosition = glm::vec3(15.f, 15.f, 50.f);
		this->worldUp = glm::vec3(0.f, 1.f, 0.f);
		this->camFront = glm::vec3(0.f, 0.f, -1.f);
		camera = new Camera(this->camPosition, this->camFront, this->worldUp);

		this->fov = 90.f;
		this->nearPlane = 0.1f;
		this->farPlane = 1000.f;

		this->initGLFW();
		this->initWindow(title, resizable);
		this->initGLEW();
		this->initOpenGLOptions();

		this->initShaders();
		this->initMatrices();
		this->initUniforms();
		this->initModel();

		this->initImgui();
	}

	void updateKeyboardInput()
	{
		//Program
		if (glfwGetKey(this->window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		{
			glfwSetWindowShouldClose(this->window, GLFW_TRUE);
		}

		//Camera
		if (glfwGetKey(this->window, GLFW_KEY_W) == GLFW_PRESS)
		{
			this->camera->move(this->dt, FORWARD);
		}
		if (glfwGetKey(this->window, GLFW_KEY_S) == GLFW_PRESS)
		{
			this->camera->move(this->dt, BACKWARD);
		}
		if (glfwGetKey(this->window, GLFW_KEY_A) == GLFW_PRESS)
		{
			this->camera->move(this->dt, LEFT);
		}
		if (glfwGetKey(this->window, GLFW_KEY_D) == GLFW_PRESS)
		{
			this->camera->move(this->dt, RIGHT);
		}
	}
	
	void updateInput()
	{
		glfwPollEvents();

		this->updateKeyboardInput();
		this->updateMouseInput();
		this->camera->updateInput(dt, -1, this->mouseOffsetX, this->mouseOffsetY);
	}

	void updateMouseInput()
	{
		int spaceState = glfwGetKey(window, GLFW_KEY_SPACE);

		if (spaceState == GLFW_PRESS && !spacePressed)
		{
			spacePressed = true;
			if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_NORMAL && !this->firstMouse)
			{
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
				this->firstMouse = true;
			}
			else
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);	
		}
		else if(spaceState == GLFW_RELEASE)
		{
			spacePressed = false;
			if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED)
			{
				glfwGetCursorPos(this->window, &this->mouseX, &this->mouseY);

				if (this->firstMouse)
				{
					this->lastMouseX = this->mouseX;
					this->lastMouseY = this->mouseY;
					this->firstMouse = false;
				}

				this->mouseOffsetX = this->mouseX - this->lastMouseX;
				this->mouseOffsetY = this->lastMouseY - this->mouseY;

				this->lastMouseX = this->mouseX;
				this->lastMouseY = this->mouseY;
			}
		}

		
	}

	void update()
	{
		this->updateDt();
		this->updateInput();

		this->flock->update(this->dt);
		this->model->setPositions(this->flock->boids_p);
		//this->model->setVelocities(this->flock->boids_v);
		this->model->updateInstancedVBO();
	}

	void updateDt()
	{
		this->curTime = static_cast<float>(glfwGetTime());
		this->dt = this->curTime - this->lastTime;
		this->lastTime = this->curTime;
		//std::cout << 1 / this->dt << "\n";
	}

	void initModel()
	{
		const unsigned int N = 1000;
		this->flock = new Flock(N, 30, 30);
		glm::vec3* positions = flock->boids_p;
		glm::vec3* velocities = flock->boids_v;
		this->model = new InstancedPyramid(N, positions, velocities);
		this->model->initBuffers();

		this->box = new Box(30, 30, 30);
		this->box->initBuffers();
	}

	int getWindowShouldClose()
	{
		return glfwWindowShouldClose(this->window);
	}

	void setWindowShouldClose()
	{
		glfwSetWindowShouldClose(this->window, GLFW_TRUE);
	}
	
	void initImgui()
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

		// Setup Dear ImGui style
		ImGui::StyleColorsDark();
		//ImGui::StyleColorsLight();

		// Setup Platform/Renderer backends
		ImGui_ImplGlfw_InitForOpenGL(this->window, true);
		ImGui_ImplOpenGL3_Init("#version 330");
	}

	void renderImgui()
	{
		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

			ImGuiIO& io = ImGui::GetIO();
			static float f = 0.0f;
			static int counter = 0;

			ImGui::Begin("Settings"); 
			ImGui::Text("Press space to enable/disable mouse!");
			//ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
			ImGui::InputFloat("turnFactor", &this->flock->turnFactor, 0.0f, 1.0f);
			ImGui::InputFloat("visualRange", &this->flock->visualRange, 0.0f, 1.0f);
			ImGui::InputFloat("protectedRange", &this->flock->protectedRange, 0.0f, 1.0f);
			ImGui::InputFloat("centeringFactor	", &this->flock->centeringFactor, 0.0f, 1.0f);
			ImGui::InputFloat("avoidFactor", &this->flock->avoidFactor, 0.0f, 1.0f);
			ImGui::InputFloat("matchingFactor", &this->flock->matchingFactor, 0.0f, 1.0f);
			ImGui::InputFloat("maxSpeed", &this->flock->maxSpeed, 0.0f, 1.0f);
			ImGui::InputFloat("minSpeed", &this->flock->minSpeed, 0.0f, 1.0f);
			ImGui::NewLine();
			ImGui::InputFloat("cameraSpeed", &this->camera->movementSpeed, 0.0f, 1.0f);
			ImGui::InputFloat("sensitivity", &this->camera->sensitivity, 0.0f, 1.0f);
			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
			ImGui::End();
		

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}

	void updateUniforms()
	{
		this->ViewMatrix = this->camera->getViewMatrix();
		this->shader->setVec3f(this->camera->getPosition(), "lightPos0");
		this->shader->setMat4fv(this->ViewMatrix, "ViewMatrix");
		this->boxShader->setMat4fv(this->ViewMatrix, "ViewMatrix");

		glfwGetFramebufferSize(this->window, &this->framebufferWidth, &this->framebufferHeight);
		if (this->framebufferHeight != 0)
		{
			this->ProjectionMatrix = glm::perspective(
				glm::radians(this->fov),
				static_cast<float>(this->framebufferWidth) / this->framebufferHeight,
				this->nearPlane,
				this->farPlane
			);
			this->shader->setMat4fv(this->ProjectionMatrix, "ProjectionMatrix");
			this->boxShader->setMat4fv(this->ProjectionMatrix, "ProjectionMatrix");
		}

		this->shader->setVec3f(this->camera->getPosition(), "cameraPos");
	}

	void render()
	{
		//DRAW ---
		//Clear
		// sky blue
		//glClearColor(0.53f, 0.81f, 0.92f, 1.f);
		// dark blue

		glClearColor(0.f, 0.1f, 0.2f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		//Update the uniforms
		this->updateUniforms();
		//Update the model

		this->model->render(this->shader);
		this->box->render(this->boxShader);
		renderImgui();

		
		//End Draw
		glfwSwapBuffers(window);
		glFlush();

		glBindVertexArray(0);
		glUseProgram(0);
	}
};
