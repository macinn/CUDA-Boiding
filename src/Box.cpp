#include"libs.h"
#include "Shader.h"

#pragma once
class Box
{
private:
	GLuint VAO, VBO, EBO;

	glm::vec3* vertices;
	GLuint* indices;
	const unsigned int noVerticies = 8;
	const unsigned int noIndicies = 24;

public:
	Box(unsigned width, unsigned height, unsigned depth)
	{
		vertices = new glm::vec3[noVerticies]{ 
			glm::vec3(0.f, 0.f, 0.f),
			glm::vec3(width, 0.f, 0.f),
			glm::vec3(width, height, 0.f),
			glm::vec3(0.f, height, 0.f),
			glm::vec3(0.f, 0.f, depth),
			glm::vec3(width, 0.f, depth),
			glm::vec3(width, height, depth),
			glm::vec3(0.f, height, depth)
		};
		indices = new GLuint[noIndicies]{
			0, 1, 1, 2, 2, 3, 3, 0,
			4, 5, 5, 6, 6, 7, 7, 4,
			0, 4, 1, 5, 2, 6, 3, 7
		};
	}	
	~Box()
	{
		delete[] vertices;
		delete[] indices;
	}
	void initBuffers()
	{
		// VAO
		glCreateVertexArrays(1, &this->VAO);
		glBindVertexArray(this->VAO);
		// VBO
		glGenBuffers(1, &this->VBO);
		glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
		glBufferData(GL_ARRAY_BUFFER, this->noVerticies * sizeof(glm::vec3), this->vertices, GL_STATIC_DRAW);
		// EBO
		glGenBuffers(1, &this->EBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->noIndicies * sizeof(GLuint), this->indices, GL_STATIC_DRAW);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);
		glEnableVertexAttribArray(0);
	}

	void render(Shader* shader)
	{
		// update 
		shader->use();
		glBindVertexArray(VAO);
		glDrawElements(GL_LINES, this->noIndicies, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

	}
};