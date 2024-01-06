#include"libs.h"
#include"BoidDrawer.cpp"
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/string_cast.hpp>


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

glm::mat4 createRotationMatrix(const glm::vec3& yLocal, const glm::vec3& targetVector) {
	glm::vec3 normalizedTarget = glm::normalize(targetVector);
	glm::vec3 axis = glm::cross(yLocal, normalizedTarget);
	float dotProductValue = glm::dot(yLocal, normalizedTarget);
	float angle = glm::acos(dotProductValue);
	glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), angle, axis);
	return rotationMatrix;
}

void updateInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
}

void framebuffer_resize_callback(GLFWwindow* window, int fbW, int fbH)
{
	glViewport(0, 0, fbW, fbH);
}

void updateInput(GLFWwindow* window, glm::vec3& position, glm::vec3& rotation, glm::vec3& scale)
{
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		position.z -= 0.001f;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		position.z += 0.001f;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		position.x -= 0.001f;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		position.x += 0.001f;
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
	{
		rotation.y -= .1f;
	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
	{
		rotation.y += .1f;
	}
	if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
	{
		scale += 0.01f;
	}
	if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
	{
		scale -= 0.01f;
	}
}


int main()
{
	//INIT GLFW
	glfwInit();

	//CREATE WINDOW
	const int WINDOW_WIDTH = 640;
	const int WINDOW_HEIGHT = 480;
	int framebufferWidth = 0;
	int framebufferHeight = 0;

	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); MAC OS

	GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Shoal of fish", NULL, NULL);

	glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
	glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);
	//IMPORTANT WHITH PERSPECTIVE MATRIX!!!

	//glViewport(0, 0, framebufferWidth, framebufferHeight);

	glfwMakeContextCurrent(window); //IMPORTANT!!

	//INIT GLEW (NEEDS WINDOW AND OPENGL CONTEXT)
	glewExperimental = GL_TRUE;

	//Error
	if (glewInit() != GLEW_OK)
	{
		std::cout << "ERROR::MAIN.CPP::GLEW_INIT_FAILED" << "\n";
		glfwTerminate();
	}

	//OPENGL OPTIONS
	glEnable(GL_DEPTH_TEST);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glFrontFace(GL_CCW);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	//SHADER INIT
	Shader core_program("vertex_core.glsl", "fragment_core.glsl");

	//MODEL

	//VAO, VBO, EBO
	//GEN VAO AND BIND
	GLuint VAO; 
	glCreateVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	//GEN VBO AND BIND AND SEND DATA
	GLuint VBO;
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	//GEN EBO AND BIND AND SEND DATA
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
	glm::vec3 velocities[] =
	{
		glm::vec3(0.001f, 0.f, 0.f),
		glm::vec3(0.001f, 0.f, 0.f),
		glm::vec3(0.001f, 0.f, 0.f)
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

	//BIND VAO 0
	glBindVertexArray(0);

	//INIT MATRICES
	glm::vec3 position(0.f);
	glm::vec3 rotation(0.f);
	glm::vec3 scale(1.f);

	glm::vec3 camPostion(0.f, 0.f, 1.f);
	glm::vec3 worldUp(0.f, 1.f, 0.f);
	glm::vec3 camFront(0.f, 0.f, -1.f);

	glm::mat4 ViewMatrix(1.f);
	ViewMatrix = glm::lookAt(camPostion, camPostion + camFront, worldUp);

	float fov = 90.f;
	float nearPlane = 0.1f;
	float farPlane = 1000.f;
	glm::mat4 ProjectionMatrix(1.f);

	ProjectionMatrix = glm::perspective(
		glm::radians(fov),
		static_cast<float>(framebufferWidth) / framebufferHeight,
		nearPlane,
		farPlane
	);

	//LIGHTS
	glm::vec3 lightPos0(0.f, 0.f, 1.f);

	//INIT UNIFORMS

	core_program.setMat4fv(ViewMatrix, "ViewMatrix");
	core_program.setMat4fv(ProjectionMatrix, "ProjectionMatrix");

	core_program.setVec3f(lightPos0, "lightPos0");
	core_program.setVec3f(camPostion, "cameraPos");
	core_program.setVec3f(glm::vec3(1.f, 0.f, 0.f), "vertex_velocity");

	//glm::mat3 rotationMatrix = createRotationMatrix(glm::vec3(0.f, 1.f, 0.f), glm::vec3(1.f, 0.f, 0.f));
	//core_program.setMat4fv(rotationMatrix, "ModelMatrix");


	//MAIN LOOP
	while (!glfwWindowShouldClose(window))
	{
		//UPDATE INPUT ---
		glfwPollEvents();
		updateInput(window, position, rotation, scale);

		//UPDATE --- 
		updateInput(window);

		//DRAW ---
		// sky blue
		//glClearColor(0.53f, 0.81f, 0.92f, 1.f);
		// dark blue
		glClearColor(0.f, 0.1f, 0.2f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		//glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);

		//ProjectionMatrix = glm::perspective(
		//	glm::radians(fov),
		//	static_cast<float>(framebufferWidth) / framebufferHeight,
		//	nearPlane,
		//	farPlane
		//);

		//core_program.setMat4fv(ProjectionMatrix, "ProjectionMatrix");

		//Use a program
		core_program.use();

		//Bind vertex array object
		glBindVertexArray(VAO);
		offsets[0].x += 0.0001f;

		glNamedBufferSubData(instanceVBO, 0, sizeof(glm::vec3) * 3, &offsets[0]);

		//Draw
		//glDrawArrays(GL_TRIANGLES, 0, nrOfVertices);
		//glDrawElements(GL_TRIANGLES, nrOfIndices, GL_UNSIGNED_INT, 0);
		glDrawElementsInstanced(GL_TRIANGLES, nrOfIndices, GL_UNSIGNED_INT, 0, 3);
		//glDrawArraysInstanced(GL_TRIANGLES, 0, 3, 3);

		//End Draw
		glfwSwapBuffers(window);
		glFlush();

		glBindVertexArray(0);
		glUseProgram(0);
	}

	//END OF PROGRAM
	glfwDestroyWindow(window);
	glfwTerminate();

	//Delete VAO and Buffers

	return 0;
}

int main2()
{
	BoidDrawer drawer("Shoal",
		1920, 1080,
		4, 4,
		false);


	//MAIN LOOP
	while (!drawer.getWindowShouldClose())
	{
		//UPDATE INPUT ---
		drawer.update();
		drawer.render();
	}

	return 0;
}

