#include"libs.h"
#include "Shader.h"
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

	//OpenGL Context
	const int GL_VERSION_MAJOR;
	const int GL_VERSION_MINOR;

	//Matrices
	glm::mat4 ViewMatrix;
	glm::vec3 camPosition;
	glm::vec3 worldUp;
	glm::vec3 camFront;

	glm::mat4 ProjectionMatrix;
	float fov;
	float nearPlane;
	float farPlane;
public:
	BoidDrawer() = default;
	~BoidDrawer() = default;

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

		//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
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
	}

	void initLight()
	{
		this->shader->setVec3f(glm::vec3(0.f, 0.f, 1.f), "lightPos0");
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

		this->camPosition = glm::vec3(0.f, 0.f, 1.f);
		this->worldUp = glm::vec3(0.f, 1.f, 0.f);
		this->camFront = glm::vec3(0.f, 0.f, -1.f);

		this->fov = 90.f;
		this->nearPlane = 0.1f;
		this->farPlane = 1000.f;

		this->initGLFW();
		this->initWindow(title, resizable);
		this->initGLEW();
		this->initOpenGLOptions();

		this->initMatrices();
		this->initShaders();
		this->initUniforms();
	}

	void updateKeyboardInput()
	{
		//Program
		if (glfwGetKey(this->window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		{
			glfwSetWindowShouldClose(this->window, GLFW_TRUE);
		}

		//Camera
		//if (glfwGetKey(this->window, GLFW_KEY_W) == GLFW_PRESS)
		//{
		//	this->camera.move(this->dt, FORWARD);
		//}
		//if (glfwGetKey(this->window, GLFW_KEY_S) == GLFW_PRESS)
		//{
		//	this->camera.move(this->dt, BACKWARD);
		//}
		//if (glfwGetKey(this->window, GLFW_KEY_A) == GLFW_PRESS)
		//{
		//	this->camera.move(this->dt, LEFT);
		//}
		//if (glfwGetKey(this->window, GLFW_KEY_D) == GLFW_PRESS)
		//{
		//	this->camera.move(this->dt, RIGHT);
		//}
		//if (glfwGetKey(this->window, GLFW_KEY_C) == GLFW_PRESS)
		//{
		//	this->camPosition.y -= 0.05f;
		//}
		//if (glfwGetKey(this->window, GLFW_KEY_SPACE) == GLFW_PRESS)
		//{
		//	this->camPosition.y += 0.05f;
		//}
	}

	void updateInput()
	{
		glfwPollEvents();

		this->updateKeyboardInput();
		//this->updateMouseInput();
		//this->camera.updateInput(dt, -1, this->mouseOffsetX, this->mouseOffsetY);
	}

	void update()
	{
		//this->updateDt();
		this->updateInput();
	}



	void bind()
	{
		const float fishH = 1.f / 2;
		const float fishW = 0.5f / 2;
		const float invSqrt3 = 1.f / sqrt(3.f);

		Vertex vertices[] =
		{
			//Position														//Normals
			glm::vec3(0.f, 0.f + fishH / 2, 0.f),							glm::vec3(0.f, 1.f, 0.f),
			glm::vec3(0.f - fishW / 2, 0.f - fishH / 2, 0.f + fishW / 2),	glm::vec3(-invSqrt3, -invSqrt3, +invSqrt3),
			glm::vec3(0.f + fishW / 2, 0.f - fishH / 2, 0.f + fishW / 2),	glm::vec3(+invSqrt3, -invSqrt3, +invSqrt3),
			glm::vec3(0.f + fishW / 2, 0.f - fishH / 2, 0.f - fishW / 2),	glm::vec3(+invSqrt3, -invSqrt3, -invSqrt3),
			glm::vec3(0.f - fishW / 2, 0.f - fishH / 2, 0.f - fishW / 2),	glm::vec3(-invSqrt3, -invSqrt3, -invSqrt3),
		};
		unsigned nrOfVertices = sizeof(vertices) / sizeof(Vertex);

		GLuint indices[] =
		{
			0, 1, 2,
			0, 2, 3,
			0, 3, 4,
			0, 4, 1,
			1, 3, 2,
			4, 3, 1
		};
		unsigned nrOfIndices = sizeof(indices) / sizeof(GLuint);

		GLuint VAO;
		glCreateVertexArrays(1, &VAO);
		glBindVertexArray(VAO);

		GLuint VBO;
		glGenBuffers(1, &VBO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		GLuint EBO;
		glGenBuffers(1, &EBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

		//Position
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, position));
		glEnableVertexAttribArray(0);
		//Normal
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, normal));
		glEnableVertexAttribArray(1);

		//Offset
		glm::vec3 offsets[] =
		{
			glm::vec3(0.f, 0.f, 0.f),
			glm::vec3(0.f, 0.5f, -15.f),
			glm::vec3(-1.f, -2.f, -2.5f)
		};

		unsigned int instanceVBO;
		glGenBuffers(1, &instanceVBO);
		glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * 3, offsets, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glEnableVertexAttribArray(2);
		glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glVertexAttribDivisor(2, 1);

		glVertexAttribDivisor(0, 0);
		glVertexAttribDivisor(1, 0);
		glVertexAttribDivisor(2, 1);

		glBindVertexArray(0);

		glBindVertexArray(VAO);

		glDrawElementsInstanced(GL_TRIANGLES, nrOfIndices, GL_UNSIGNED_INT, 0, 3);
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
		//Update the model

		this->shader->use();
		this->bind();
		
		

		//End Draw
		glfwSwapBuffers(window);
		glFlush();

		glBindVertexArray(0);
		glUseProgram(0);
	}

	int getWindowShouldClose()
	{
		return glfwWindowShouldClose(this->window);
	}

	void setWindowShouldClose()
	{
		glfwSetWindowShouldClose(this->window, GLFW_TRUE);
	}
};
