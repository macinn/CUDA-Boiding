#pragma once

#include <random>
#include <time.h>
#include <glm/ext/vector_float3.hpp>
#include <glm/gtx/norm.hpp>
#include "BoidsEngine.h"

class BoidsEngine_CPU: public BoidsEngine {
protected:
	// Boids position and velocity
	glm::vec3* boids_p;
	glm::vec3* boids_v;

	// Initialize boids position
	void init() override {
		std::default_random_engine rd{ static_cast<unsigned int>(time(0)) };
		std::mt19937 gen{ rd() };
		std::uniform_real_distribution<> w(0, width);
		std::uniform_real_distribution<> h(0, height);
		std::uniform_real_distribution<> z(0, depth);

		for (unsigned int i = 0; i < N; i++) {
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
	virtual void updateData(float dt)
	{
		float visualRangeSquared = visualRange * visualRange;
		float protectedRangeSquared = protectedRange * protectedRange;

		for (unsigned int i = 0; i < N; i++) {

			unsigned int countVisible = 0;
			unsigned int countClose = 0;
			glm::vec3 vel = glm::vec3(0.0f);
			glm::vec3 center = glm::vec3(0.0f);
			glm::vec3 close = glm::vec3(0.0f);

			for (unsigned int j = 0; j < N; j++) {
				if (i != j) {
					float distanceSquared = glm::length2(boids_p[i] - boids_p[j]);
					if (distanceSquared < visualRangeSquared)
					{
						center += boids_p[j];
						countVisible++;

						if (distanceSquared < protectedRangeSquared)
						{
							vel += boids_v[j];
							close += (boids_p[i] - boids_p[j])
								* (protectedRangeSquared - distanceSquared);
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

			close /= protectedRangeSquared;
			boids_v[i] +=
				(center - boids_p[i]) * centeringFactor		// cohesion
				+ close * avoidFactor						// separation	
				+ (vel - boids_v[i]) * matchingFactor;		// alignment
		}
		for (int i = 0; i < N; i++)
		{
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
	BoidsEngine_CPU(unsigned int N, unsigned int width, unsigned int height, unsigned int depth):
		BoidsEngine(N, width, height, depth) {}
	

	virtual ~BoidsEngine_CPU() {
		delete[] boids_p;
		delete[] boids_v;
	}

	// Update boids position and velocity
	virtual void update(float dt, GLuint positionBuffer, GLuint velocityBuffer) {
		updateData(dt);
		updateBuffers(positionBuffer, velocityBuffer);
	}

	// Visual range setter
	virtual void setVisualRange(float visualRange) {
		this->visualRange = visualRange;
	}
};

