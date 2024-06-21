#pragma once

#include "BoidsEngine_CPU.cpp"
#include <glm/gtx/norm.hpp>

class BoidsLogicTEST : public BoidsEngine_CPU {
protected:
	void updateData(float dt) override
	{
		{
			float visualRangeSquared = visualRange * visualRange;
			float protectedRangeSquared = protectedRange * protectedRange;
			for (int i = 0; i < N; i++) {

				unsigned int countVisible = 0;
				unsigned int countClose = 0;
				glm::vec3 vel = glm::vec3(0.0f);
				glm::vec3 center = glm::vec3(0.0f);
				glm::vec3 close = glm::vec3(0.0f);

				for (unsigned int j = 0; j < N; j++) {
					glm::vec3 diff = boids_p[i] - boids_p[j];
					if (/*diff.x < visualRange 
						&& diff.y < visualRange 
						&& diff.z < visualRange
						&&*/ i != j) {
						float distanceSquared = glm::length2(diff);
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
		}
		for (int i = 0; i < N; i++)
		{
			boundPosition(i);
			boundVelocity(i);
			boids_p[i] += boids_v[i] * dt;
		}
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
	BoidsLogicTEST(unsigned int N, unsigned int width, unsigned int height, unsigned int depth) : BoidsEngine_CPU(N, width, height, depth) {}

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

