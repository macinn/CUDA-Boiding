#pragma once

#include <GL/glew.h>

class BoidsEngine
{
protected:
	// Boids number and container size
	const unsigned int N;
	const unsigned int width;
	const unsigned int height;
	const unsigned int depth;

	// Margin factor for boids to turn around
	const float marginFactor = 0.05f;

	// Allocate memory for boids position and velocity
	virtual void alloc() {};

	// Initialize boids positioninline
	virtual void init() {};

	// Bound boid velocity to min and max speed
	void inline boundVelocity(int i) = delete;

	// Bound boid position to container
	void inline boundPosition(int i) = delete;

	// Update position and velocity
	virtual void updateData(float dt) {};

	// Update data in model buffors
	virtual void updateBuffers(GLuint positionBuffer, GLuint velocityBuffer) {};

public:
	// Movement parameters
	float turnFactor = 0.2f;
	float visualRange = 3.f;
	float protectedRange = 0.75f;
	float centeringFactor = 0.03f;
	float avoidFactor = 0.05f;
	float matchingFactor = 0.1f;
	float maxSpeed = 10.f;
	float minSpeed = 5.f;

	// Constructor and destructor
	BoidsEngine(unsigned int N, unsigned int width, unsigned int height, unsigned int depth) :
		N(N), width(width), height(height), depth(!depth ? (width + height) / 2 : depth) {
		this->alloc();
		this->init();
	};

	virtual ~BoidsEngine() {};

	// Update boids position and velocity
	virtual void update(float dt, GLuint positionBuffer, GLuint velocityBuffer) {
		updateData(dt);
		updateBuffers(positionBuffer, velocityBuffer);
	}

	// Visual range setter
	virtual void setVisualRange(float visualRange) = 0;
};