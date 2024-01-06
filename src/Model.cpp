#include"libs.h"
#include "Shader.h"

#pragma once
class InstancedPyramid
{
private:
	GLuint VAO, VBO, EBO;
	GLuint instancePosition;
	GLuint instanceVelocity;

	Vertex* vertices;
	GLuint* indices;
	const unsigned int noVerticies = 5;
	const unsigned int noIndicies = 18;

	glm::vec3* positions;
	glm::vec3* velocities;
	const unsigned int N;
public:
	InstancedPyramid(unsigned N, glm::vec3* positions, glm::vec3* velocities) :N(N), positions(positions), velocities(velocities)
	{
		const float fishH = 1.f / 2;
		const float fishW = 0.5f / 2;
		const float invSqrt3 = 1.f / sqrt(3.f);

		vertices = new Vertex[noVerticies]
		{
			//Position														//Normals
			glm::vec3(0.f, 0.f + fishH / 2, 0.f),							glm::vec3(0.f, 1.f, 0.f),
			glm::vec3(0.f - fishW / 2, 0.f - fishH / 2, 0.f + fishW / 2),	glm::vec3(-invSqrt3, -invSqrt3, +invSqrt3),
			glm::vec3(0.f + fishW / 2, 0.f - fishH / 2, 0.f + fishW / 2),	glm::vec3(+invSqrt3, -invSqrt3, +invSqrt3),
			glm::vec3(0.f + fishW / 2, 0.f - fishH / 2, 0.f - fishW / 2),	glm::vec3(+invSqrt3, -invSqrt3, -invSqrt3),
			glm::vec3(0.f - fishW / 2, 0.f - fishH / 2, 0.f - fishW / 2),	glm::vec3(-invSqrt3, -invSqrt3, -invSqrt3),
		};

		// Triangle indices
		indices = new GLuint[noIndicies]
		{
			0, 1, 2,
			0, 2, 3,
			0, 3, 4,
			0, 4, 1,
			1, 3, 2,
			4, 3, 1
		};
	}
	~InstancedPyramid()
	{
		delete[] vertices;
		delete[] indices;
		delete[] positions;
		delete[] velocities;
	}
	void initBuffers()
	{
		// VAO
		glCreateVertexArrays(1, &this->VAO);
		glBindVertexArray(this->VAO);
		// VBO
		glGenBuffers(1, &this->VBO);
		glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
		glBufferData(GL_ARRAY_BUFFER, this->noVerticies * sizeof(Vertex), this->vertices, GL_STATIC_DRAW);
		// EBO
		glGenBuffers(1, &this->EBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->noIndicies * sizeof(GLuint), this->indices, GL_STATIC_DRAW);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, position));
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, normal));
		glEnableVertexAttribArray(1);

		// instancePosition
		glGenBuffers(1, &instancePosition);
		glBindBuffer(GL_ARRAY_BUFFER, instancePosition);
		glBufferData(GL_ARRAY_BUFFER, this->N * sizeof(glm::vec3), this->positions, GL_STATIC_DRAW);

		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
		glEnableVertexAttribArray(2);
		glVertexAttribDivisor(2, 1);
		// instancePosition
		glGenBuffers(1, &instanceVelocity);
		glBindBuffer(GL_ARRAY_BUFFER, instanceVelocity);
		glBufferData(GL_ARRAY_BUFFER, this->N * sizeof(glm::vec3), this->velocities, GL_STATIC_DRAW);

		glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
		glEnableVertexAttribArray(3);
		glVertexAttribDivisor(3, 1);
	}
	void updateInstancedVBO()
	{

	}
	void render(Shader* shader)
	{
		// update 
		shader->use();

	}
};