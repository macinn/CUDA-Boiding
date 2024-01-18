#include <GL/glew.h>
#include <glm/vec3.hpp>
#include <glm/gtx/norm.hpp>
#include <random>
#define uint unsigned int

#pragma once

class BoidsLogic {
protected:
	// Boids number and container size
	const uint N;
	const uint width;
	const uint height;
	const uint depth;

	// Boids position and velocity
	glm::vec3* boids_p;
	glm::vec3* boids_v;

	// Margin factor for boids to turn around
	const float marginFactor = 0.05f;

	// Initialize boids position
	void init() {
		std::default_random_engine rd{ static_cast<long uint>(time(0)) };
		std::mt19937 gen{ rd() };
		std::uniform_real_distribution<> w(0, width);
		std::uniform_real_distribution<> h(0, height);
		std::uniform_real_distribution<> z(0, depth);

		for (uint i = 0; i < N; i++) {
			boids_p[i] = glm::vec3(w(gen), h(gen), z(gen));
		}
	}

	// Bound boid velocity to min and max speed
	void boundVelocity(int i) {
		float speed = glm::l2Norm(boids_v[i]);
		if (speed > maxSpeed) {
			boids_v[i] /= speed;
			boids_v[i] *= maxSpeed;
		}
		else if (speed < minSpeed) {
			boids_v[i] /= speed;
			boids_v[i] *= minSpeed;
		}
	}

	// Bound boid position to container
	void boundPosition(int i) {
		if (boids_p[i].x < width * marginFactor) {
			boids_v[i].x += turnFactor;
		}
		if (boids_p[i].x > width * (1 - marginFactor)) {
			boids_v[i].x -= turnFactor;
		}
		if (boids_p[i].y < height * marginFactor) {
			boids_v[i].y += turnFactor;
		}
		if (boids_p[i].y > height * (1 - marginFactor)) {
			boids_v[i].y -= turnFactor;
		}
		if (boids_p[i].z < depth * marginFactor) {
			boids_v[i].z += turnFactor;
		}
		if (boids_p[i].z > depth * (1 - marginFactor)) {
			boids_v[i].z -= turnFactor;
		}
	}

	// Update position and velocity
	void updateData(float dt)
	{
		float visualRangeSquared = visualRange * visualRange;
		float protectedRangeSquared = protectedRange * protectedRange;

		for (uint i = 0; i < N; i++) {

			uint countVisible = 0;
			uint countClose = 0;
			glm::vec3 vel = glm::vec3(0.0f);
			glm::vec3 center = glm::vec3(0.0f);
			glm::vec3 close = glm::vec3(0.0f);

			for (uint j = 0; j < N; j++) {
				if (i != j) {
					float distanceSquared = glm::distance2(boids_p[i], boids_p[j]);
					if (distanceSquared < visualRangeSquared)
					{
						center += boids_p[j];
						countVisible++;

						if (distanceSquared < protectedRangeSquared)
						{
							vel += boids_v[j];
							close -= boids_p[j];
							countClose++;
						}
					}

				}
			}

			if (countVisible > 0) {
				center /= countVisible;

				if (countClose > 0) {
					vel /= countClose;
				}
			}

			close += (float)countClose * boids_p[i];
			boids_v[i] +=
				(center - boids_p[i]) * centeringFactor		// cohesion
				+ close * avoidFactor						// separation	
				+ (vel - boids_v[i]) * matchingFactor;		// alignment

			boundPosition(i);
			boundVelocity(i);
			boids_p[i] += boids_v[i] * dt;
		}
	}

	// Update data in model buffors
	void updateBuffers(GLuint positionBuffer, GLuint velocityBuffer)
	{
		glNamedBufferSubData(positionBuffer, 0, this->N * sizeof(glm::vec3), this->boids_p);
		glNamedBufferSubData(velocityBuffer, 0, this->N * sizeof(glm::vec3), this->boids_v);
	}

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
	BoidsLogic(uint N, uint width, uint height, uint depth = 0):
		N(N), width(width), height(height), depth(!depth ? (width+height)/2 : depth)
	{
		boids_p = new glm::vec3[N]();
		boids_v = new glm::vec3[N]();
		this->init();
	}
	virtual ~BoidsLogic() {
		delete[] boids_p;
		delete[] boids_v;
	}

	// Update boids position and velocity
	virtual void update(float dt, GLuint positionBuffer, GLuint velocityBuffer) {
		updateData(dt);
		updateBuffers(positionBuffer, velocityBuffer);
	}
};

