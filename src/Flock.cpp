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

public:
	// Parameters
	float turnFactor = 0.2f;
	float visualRange = 8.f;
	float protectedRange = 1.f;
	float centeringFactor = 0.0005f;
	float avoidFactor = 0.05f;
	float matchingFactor = 0.1f;
	float maxSpeed = 10.f;
	float minSpeed = 5.f;
	const float marginFactor = 0.f;

	// Boids
	glm::vec3* boids_p;
	glm::vec3* boids_v;
	Flock(uint N, uint width, uint height, uint depth = 0): 
		N(N), width(width), height(height), depth(!depth ? (width+height)/2 : depth)
	{
		boids_p = new glm::vec3[N]();
		boids_v = new glm::vec3[N]();
	}
	~Flock() {
		delete[] boids_p;
		delete[] boids_v;
	}
	void update(float dt) {
		for (uint i = 0; i < N; i++) {
			boids_v[i] += cohesion(i) + separation(i) + alignment(i);
			boundPosition(i);
			boundVelocity(i);

			boids_p[i] += boids_v[i] * dt;
		}
	}
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
	void init() {
		std::default_random_engine rd{ static_cast<long unsigned int>(time(0)) };
		std::mt19937 gen{ rd() };
		std::uniform_real_distribution<> w(0, width);
		std::uniform_real_distribution<> h(0, height);
		std::uniform_real_distribution<> z(0, depth);

		float boxSize = 2 * protectedRange;
		uint indexStride = (width - 1)/boxSize + 1;
		for (uint i = 0; i < N; i++) {
			boids_p[i] = glm::vec3(w(gen), h(gen), z(gen));
		}
	}
	glm::vec3 cohesion(int i) {
		glm::vec3 center = glm::vec3(0.0f);
		unsigned int count = 0;
		for (int j = 0; j < N; j++) {
			if (i != j) {
				if (glm::distance(boids_p[i], boids_p[j]) < visualRange) {
					center += boids_p[j];
					count++;
				}
			}
		}	
		if (count > 0) {
			center /= count;
		}
		return (center - boids_p[i])*centeringFactor;
	}
	glm::vec3 separation(int i) {
		glm::vec3 close = glm::vec3(0.0f);
		for (int j = 0; j < N; j++) {
			if (i != j) {
				if (glm::distance(boids_p[i], boids_p[j]) < protectedRange) {
					close += boids_p[i] - boids_p[j];
				}
			}
		}
		return close;
	}
	glm::vec3 alignment(int i) {
		unsigned int count = 0;
		glm::vec3 vel = glm::vec3(0.0f);
		for (int j = 0; j < N; j++) {
			if (i != j) {
				if (glm::distance(boids_p[i], boids_p[j]) < protectedRange) {
					count++;
					vel += boids_v[j];
				}
			}
		}
		if (count > 0) {
			vel /= count;
		}
		return (vel - boids_v[i])*matchingFactor;
	}
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
};

