#include "BoidsLogic.cpp"
#include "BoidsModel.cpp"
#include "Camera.cpp"
#include "BoxModel.cpp"

#include "backends/imgui_impl_opengl3.h"
#include "backends/imgui_impl_glfw.h"

#pragma once

// Main drawer class
class BoidsDrawer
{
private:
	// OpenGL window paremeters
	GLFWwindow* window;
	const int WINDOW_WIDTH;
	const int WINDOW_HEIGHT;
	int framebufferWidth;
	int framebufferHeight;
	const int GL_VERSION_MAJOR;
	const int GL_VERSION_MINOR;
	bool updateModels = true;
	
	// Models and shaders
	Shader* boidsShader;
	BoidsModel* boidsModel;
	Shader* boxShader;
	BoxModel* boxModel;

	// Boids logic
	BoidsLogic* boidsLogic;

	// Camera
	Camera* camera;
	const float fov = 90.f;
	const float nearPlane = 0.1f;
	const float farPlane = 1000.f;

	// Matrices
	glm::mat4 ViewMatrix;
	glm::vec3 camPosition;
	glm::vec3 worldUp;
	glm::vec3 camFront;
	glm::mat4 ProjectionMatrix;

	// Delta time
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

	// Create window
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
		glfwSetFramebufferSizeCallback(window, BoidsDrawer::framebuffer_resize_callback);

		glfwMakeContextCurrent(this->window);
	}

	// Window resize callback
	static void framebuffer_resize_callback(GLFWwindow* window, int fbW, int fbH)
	{
		glViewport(0, 0, fbW, fbH);
	}

	// Initialize GLFW
	void initGLFW()
	{
		if (glfwInit() == GLFW_FALSE)
		{
			std::cout << "ERROR::GLFW_INIT_FAILED" << "\n";
			glfwTerminate();
		}
	}

	// Initialize GLEW
	void initGLEW()
	{
		glewExperimental = GL_TRUE;

		if (glewInit() != GLEW_OK)
		{
			std::cout << "ERROR::MAIN.CPP::GLEW_INIT_FAILED" << "\n";
			glfwTerminate();
		}
	}

	// Initialize standard OpenGL options
	void initOpenGLOptions()
	{
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
		glFrontFace(GL_CCW);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		// glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}

	// Initialize view and projection matrices
	void initMatrices()
	{
		this->ViewMatrix = glm::lookAt(this->camPosition, this->camPosition + this->camFront, this->worldUp);

		this->ProjectionMatrix = glm::perspective(
			glm::radians(this->fov),
			static_cast<float>(this->framebufferWidth) / this->framebufferHeight,
			this->nearPlane,
			this->farPlane
		);
	}

	// Initialize shaders
	void initShaders()
	{
		this->boidsShader = new Shader("shaders/boids/vertex_core.glsl", "shaders/boids/fragment_core.glsl");
		this->boxShader = new Shader("shaders/box/vertex_box.glsl", "shaders/box/fragment_box.glsl");
	}

	// Initialize viewMatrix, projectionMatrix and light
	void initUniforms()
	{
		this->boidsShader->setMat4fv(ViewMatrix, "ViewMatrix");
		this->boidsShader->setMat4fv(ProjectionMatrix, "ProjectionMatrix");
		this->boidsShader->setVec3f(this->camPosition, "lightPos0");
	}

	// Update camera based on user input
	void updateInput()
	{
		glfwPollEvents();

		this->updateKeyboardInput();
		this->updateMouseInput();
		this->camera->updateMouseInput(dt, this->mouseOffsetX, this->mouseOffsetY);
	}

	// Update keyes pressed
	void updateKeyboardInput()
	{
		// Close window on ESC
		if (glfwGetKey(this->window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		{
			glfwSetWindowShouldClose(this->window, GLFW_TRUE);
		}

		// Move camera
		if (glfwGetKey(this->window, GLFW_KEY_W) == GLFW_PRESS)
		{
			this->camera->updateKeyboardInput(this->dt, FORWARD);
		}
		if (glfwGetKey(this->window, GLFW_KEY_S) == GLFW_PRESS)
		{
			this->camera->updateKeyboardInput(this->dt, BACKWARD);
		}
		if (glfwGetKey(this->window, GLFW_KEY_A) == GLFW_PRESS)
		{
			this->camera->updateKeyboardInput(this->dt, LEFT);
		}
		if (glfwGetKey(this->window, GLFW_KEY_D) == GLFW_PRESS)
		{
			this->camera->updateKeyboardInput(this->dt, RIGHT);
		}
	}

	// Update mouse input
	void updateMouseInput()
	{
		// So we count space press only once
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
		else if (spaceState == GLFW_RELEASE)
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

	// Update delta time
	void updateDt()
	{
		this->curTime = static_cast<float>(glfwGetTime());
		this->dt = this->curTime - this->lastTime;
		this->lastTime = this->curTime;
	}

	// Initialize boids and box models
	void initModels(uint N, uint size)
	{
		this->boidsLogic = new BoidsLogic(N, size, size, size);

		this->boidsModel = new BoidsModel(N,
			this->boidsLogic->boids_p, this->boidsLogic->boids_v);

		this->boxModel = new BoxModel(size, size, size);
	}

	// Initialize user interface
	void initImgui()
	{
		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

		// Setup Dear ImGui style
		ImGui::StyleColorsDark();

		// Setup Platform/Renderer backends
		ImGui_ImplGlfw_InitForOpenGL(this->window, true);
		ImGui_ImplOpenGL3_Init("#version 330");
	}

	// Render user interface
	void renderImgui()
	{
		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGuiIO& io = ImGui::GetIO();

		ImGui::Begin("Settings");
		ImGui::Text("Press SPACE to enable/disable mouse.");
		ImGui::Text("Press ESC to close window.");

		if (ImGui::Button("Stop animation"))
		{
			this->updateModels = !this->updateModels;
		}
		ImGui::InputFloat("turnFactor", &this->boidsLogic->turnFactor, 0.0f, 1.0f);
		ImGui::InputFloat("visualRange", &this->boidsLogic->visualRange, 0.0f, 1.0f);
		ImGui::InputFloat("protectedRange", &this->boidsLogic->protectedRange, 0.0f, 1.0f);
		ImGui::InputFloat("centeringFactor	", &this->boidsLogic->centeringFactor, 0.0f, 1.0f);
		ImGui::InputFloat("avoidFactor", &this->boidsLogic->avoidFactor, 0.0f, 1.0f);
		ImGui::InputFloat("matchingFactor", &this->boidsLogic->matchingFactor, 0.0f, 1.0f);
		ImGui::InputFloat("maxSpeed", &this->boidsLogic->maxSpeed, 0.0f, 1.0f);
		ImGui::InputFloat("minSpeed", &this->boidsLogic->minSpeed, 0.0f, 1.0f);
		ImGui::NewLine();
		ImGui::InputFloat("cameraSpeed", &this->camera->movementSpeed, 0.0f, 1.0f);
		ImGui::InputFloat("sensitivity", &this->camera->sensitivity, 0.0f, 1.0f);
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
		ImGui::End();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
public:
	// Constructor and destructor
	BoidsDrawer(const char* title,
		const int WINDOW_WIDTH, const int WINDOW_HEIGHT,
		const int GL_VERSION_MAJOR, const int GL_VERSION_MINOR,
		bool resizable, uint N, uint size
	) :WINDOW_WIDTH(WINDOW_WIDTH),
		WINDOW_HEIGHT(WINDOW_HEIGHT),
		GL_VERSION_MAJOR(GL_VERSION_MAJOR),
		GL_VERSION_MINOR(GL_VERSION_MINOR)
	{
		this->window = nullptr;
		this->framebufferWidth = this->WINDOW_WIDTH;
		this->framebufferHeight = this->WINDOW_HEIGHT;

		this->camPosition = glm::vec3( size / 2, size / 2, 2 * size);
		this->worldUp = glm::vec3(0.f, 1.f, 0.f);
		this->camFront = glm::vec3(0.f, 0.f, -1.f);

		camera = new Camera(this->camPosition, this->camFront, this->worldUp);

		this->initGLFW();
		this->initWindow(title, resizable);
		this->initGLEW();
		this->initOpenGLOptions();

		this->initShaders();
		this->initMatrices();
		this->initUniforms();
		this->initModels(N, size);

		this->initImgui();
	}
	~BoidsDrawer()
	{
		glfwDestroyWindow(this->window);
		glfwTerminate();

		delete this->boidsShader;
		delete this->boxShader;
		delete this->camera;
		delete this->boidsModel;
		delete this->boidsLogic;
		delete this->boxModel;
	}

	// Update objects
	void update()
	{
		this->updateDt();
		this->updateInput();
		if(updateModels)
			this->boidsLogic->update(this->dt);
	}

	// Check if window should close
	int getWindowShouldClose()
	{
		return glfwWindowShouldClose(this->window);
	}

	// Set window should close
	void setWindowShouldClose()
	{
		glfwSetWindowShouldClose(this->window, GLFW_TRUE);
	}
	
	// Update uniforms
	void updateUniforms()
	{
		this->ViewMatrix = this->camera->getViewMatrix();
		
		// Update light position to camera position
		this->boidsShader->setVec3f(this->camera->getPosition(), "lightPos0");

		// Update view matrix
		this->boidsShader->setMat4fv(this->ViewMatrix, "ViewMatrix");
		this->boxShader->setMat4fv(this->ViewMatrix, "ViewMatrix");

		// Update projection matrix, if window is not minimized
		glfwGetFramebufferSize(this->window, &this->framebufferWidth, &this->framebufferHeight);
		if (this->framebufferHeight != 0)
		{
			this->ProjectionMatrix = glm::perspective(
				glm::radians(this->fov),
				static_cast<float>(this->framebufferWidth) / this->framebufferHeight,
				this->nearPlane,
				this->farPlane
			);
			this->boidsShader->setMat4fv(this->ProjectionMatrix, "ProjectionMatrix");
			this->boxShader->setMat4fv(this->ProjectionMatrix, "ProjectionMatrix");
		}
	}

	// Render new frame
	void render()
	{
		// Background color: sky blue
		//glClearColor(0.53f, 0.81f, 0.92f, 1.f);
		// 
		// Background color: dark blue
		glClearColor(0.f, 0.1f, 0.2f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		renderImgui();

		// Update the uniforms
		this->updateUniforms();

		// Render models
		this->boidsModel->render(this->boidsShader);
		this->boxModel->render(this->boxShader);
	
		// End Draw
		glfwSwapBuffers(window);
		glFlush();
		glBindVertexArray(0);
		glUseProgram(0);
	}
};
