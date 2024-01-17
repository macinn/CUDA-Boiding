#include "Shader.cpp"
#include "Model.h"

#pragma once

// Structure of vertex with normal vector and position
struct Vertex
{
	glm::vec3 position;
	glm::vec3 normal;
};

class BoidsModel
{
private:
	// OpenGL buffers, same for every instance
	GLuint VAO, VBO, EBO;
	// OpenGL buffers, instance specific
	GLuint instancePosition;
	GLuint instanceVelocity;

	// Vertices and indices
	Vertex* vertices;
	GLuint* indices;
	const unsigned int noVerticies = BOID_NO_VERTICES;
	const unsigned int noIndicies = BOID_NO_INDICES;
	
	// IInstance specific data pointers
	glm::vec3* positions;
	glm::vec3* velocities;
	// Number of boids
	const unsigned int N;

	// Initialize buffers
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
		glBufferData(GL_ARRAY_BUFFER, this->N * sizeof(glm::vec3), this->positions, GL_DYNAMIC_DRAW);

		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
		glEnableVertexAttribArray(2);
		glVertexAttribDivisor(2, 1);
		// instancePosition
		glGenBuffers(1, &instanceVelocity);
		glBindBuffer(GL_ARRAY_BUFFER, instanceVelocity);
		glBufferData(GL_ARRAY_BUFFER, this->N * sizeof(glm::vec3), this->velocities, GL_DYNAMIC_DRAW);

		glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
		glEnableVertexAttribArray(3);
		glVertexAttribDivisor(3, 1);
	}

	// Update instance specific data on GPU
	void updateInstancedVBO()
	{
		glNamedBufferSubData(instancePosition, 0, this->N * sizeof(glm::vec3), this->positions);
		glNamedBufferSubData(instanceVelocity, 0, this->N * sizeof(glm::vec3), this->velocities);
	}
public:
	// Constructor and destructor
	BoidsModel(unsigned N, glm::vec3* positions, glm::vec3* velocities) :N(N), positions(positions), velocities(velocities)
	{
		vertices = new Vertex[noVerticies]{ BOID_VERTICES };
		indices = new GLuint[noIndicies]{ BOID_INDICES };
		this->initBuffers();
	}
	~BoidsModel()
	{
		delete[] vertices;
		delete[] indices;
		glDeleteVertexArrays(1, &this->VAO);
		glDeleteVertexArrays(1, &this->instancePosition);
		glDeleteVertexArrays(1, &this->instanceVelocity);
		glDeleteBuffers(1, &this->VBO);
		glDeleteBuffers(1, &this->EBO);
	}

	// Render boids using given shader
	void render(Shader* shader)
	{
		shader->use();
		glBindVertexArray(VAO);
		updateInstancedVBO();
		glDrawElementsInstanced(GL_TRIANGLES, noIndicies, GL_UNSIGNED_INT, 0, N);
		glBindVertexArray(0);

	}

	// Setters
	void setPositions(glm::vec3* positions)
	{
		this->positions = positions;
	}

	void setVelocities(glm::vec3* velocities)
	{
		this->velocities = velocities;
	}
};