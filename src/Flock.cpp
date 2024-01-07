#include <glm/vec3.hpp>
#include <glm/gtx/norm.hpp>
#include <random>
#define uint unsigned int

class Flock {
private:
	const uint N;
	const uint width;
	const uint height;
	const uint depth;

	// Boids
public:
	glm::vec3* boids_p;
	glm::vec3* boids_v;
	Flock(uint N, uint width, uint height): N(N), width(width), height(height), depth((width+height)/2)
	{
		boids_p = new glm::vec3[N]();
		boids_v = new glm::vec3[N]();
	}

	~Flock() {
		delete[] boids_p;
		delete[] boids_v;
	}

	void update() {
		for (uint i = 0; i < N; i++) {
			boids_v[i] += cohesion(i) + separation(i) + alignment(i);
			boids_p[i] += boids_v[i];
			boundPosition(i);
		}
	}

	void init() {
		std::random_device rd{};
		std::mt19937 gen{ rd() };
		std::uniform_real_distribution<> w(0, width);
		std::uniform_real_distribution<> h(0, height);
		std::uniform_real_distribution<> z(0, depth);

		float boxSize = 2 * separationCoefficient;
		uint indexStride = (width - 1)/boxSize + 1;
		for (uint i = 0; i < N; i++) {
			boids_p[i] = glm::vec3(w(gen), h(gen), z(gen));
		}
	}

	glm::vec3 getCenter() {
		glm::vec3 center = glm::vec3(0.0f);
		for (uint i = 0; i < N; i++) {
			center += boids_p[i];
		}
		return center;
	}

	const float cohesionCoefficient = 0.01f;
	glm::vec3 cohesion(int i) {
		glm::vec3 center = getCenter();
		glm::vec3 direction = center - boids_p[i];
		return direction * cohesionCoefficient;
	}

	float separationCoefficient = 100.f;
	glm::vec3 separation(int i) {
		glm::vec3 direction = glm::vec3(0.0f);
		for (uint j = 0; j < N; j++)
		{
			if (i != j && glm::l2Norm(boids_p[i]-boids_p[j]) < separationCoefficient)
			{
				direction -= boids_p[i] - boids_p[j];
			}
		}
		return direction;
	}

	const float alignmentCoefficient = 0.125f;
	glm::vec3 alignment(int i) {
		glm::vec3 averageVelocity = glm::vec3(0.0f);
		for (uint j = 0; j < N; j++) {
			if (i != j) {
				averageVelocity += boids_v[i];
			}
		}
		averageVelocity /= N - 1;
		return (averageVelocity - boids_v[i]) * alignmentCoefficient;
	}

	float boundCoefficient = 10.f;
	void boundPosition(int i) {
		if (boids_p[i].x < 0) {
			boids_v[i].x += boundCoefficient;
		}
		else if (boids_p[i].x > width) {
			boids_v[i].x -= boundCoefficient;
		}
		if (boids_p[i].y < 0) {
			boids_v[i].y += boundCoefficient;
		}
		else if (boids_p[i].y > height) {
			boids_v[i].y -= boundCoefficient;
		}
		if (boids_p[i].z < 0) {
			boids_v[i].z += boundCoefficient;
		}
		else if (boids_p[i].z > depth) {
			boids_v[i].z -= boundCoefficient;
		}
	}	
};

