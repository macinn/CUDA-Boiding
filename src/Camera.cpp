#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#pragma once

enum direction { FORWARD = 0, BACKWARD, LEFT, RIGHT };

class Camera
{
private:
	glm::mat4 ViewMatrix;

	// Position varaibles
	glm::vec3 worldUp;
	glm::vec3 position;
	glm::vec3 front;
	glm::vec3 right;
	glm::vec3 up;

	// Euler angles
	GLfloat rotationX;
	GLfloat rotationY;
	GLfloat rotationZ;

	// Updates camera orientation vectors
	void updateCameraVectors()
	{
		this->front.x = cos(glm::radians(this->rotationZ)) * cos(glm::radians(this->rotationX));
		this->front.y = sin(glm::radians(this->rotationX));
		this->front.z = sin(glm::radians(this->rotationZ)) * cos(glm::radians(this->rotationX));

		this->front = glm::normalize(this->front);
		this->right = glm::normalize(glm::cross(this->front, this->worldUp));
		this->up = glm::normalize(glm::cross(this->right, this->front));
	}

public:
	// Movement variables
	GLfloat movementSpeed;
	GLfloat sensitivity;

	// Constructor
	Camera(glm::vec3 position, glm::vec3 direction, glm::vec3 worldUp)
	{
		this->ViewMatrix = glm::mat4(1.f);

		this->movementSpeed = 8.f;
		this->sensitivity = 2.f;

		this->worldUp = worldUp;
		this->position = position;
		this->right = glm::vec3(0.f);
		this->up = worldUp;

		this->rotationX = 0.f;
		this->rotationZ = 0.f;
		this->rotationY = 0.f;

		this->updateCameraVectors();
	}

	~Camera() {}

	// Update and return view matrix
	const glm::mat4 getViewMatrix()
	{
		this->updateCameraVectors();

		this->ViewMatrix = glm::lookAt(this->position, this->position + this->front, this->up);

		return this->ViewMatrix;
	}

	// Get camera position
	const glm::vec3 getPosition() const
	{
		return this->position;
	}

	// Update position
	void updateKeyboardInput(const float& dt, const int direction)
	{
		//Update position vector
		switch (direction)
		{
		case FORWARD:
			this->position += this->front * this->movementSpeed * dt;
			break;
		case BACKWARD:
			this->position -= this->front * this->movementSpeed * dt;
			break;
		case LEFT:
			this->position -= this->right * this->movementSpeed * dt;
			break;
		case RIGHT:
			this->position += this->right * this->movementSpeed * dt;
			break;
		default:
			break;
		}
	}

	// Update rotation
	void updateMouseInput(const float& dt, const double& offsetX, const double& offsetY)
	{
		this->rotationX += (GLfloat)offsetY * this->sensitivity * dt;
		this->rotationZ += (GLfloat)offsetX * this->sensitivity * dt;

		if (this->rotationX > 80.f)
			this->rotationX = 80.f;
		else if (this->rotationX < -80.f)
			this->rotationX = -80.f;

		if (this->rotationZ > 360.f || this->rotationZ < -360.f)
			this->rotationZ = 0.f;
	}
};