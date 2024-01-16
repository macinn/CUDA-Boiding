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

	uint* boids_grid_id;
	float grid_size = 2 * visualRange;
	uint grid_size_x = (width - 1) / grid_size + 1;
	uint grid_size_y = (height - 1) / grid_size + 1;
	uint grid_size_z = (depth - 1) / grid_size + 1;

	uint getGridId(uint x, uint y, uint z) {
		return x + y * grid_size_x + z * grid_size_x * grid_size_y;
	}
	uint getGridId(glm::vec3 pos) {
		return getGridId(pos.x / grid_size, pos.y / grid_size, pos.z / grid_size);
	}
	void updateGrid() {
		for (uint i = 0; i < N; i++) {
			boids_grid_id[i] = getGridId(boids_p[i]);
		}	
	}
	bool isNeighbor(uint x, uint y, uint z) {
		const uint x_begin = std::max(x - 1, 0u);
		const uint x_end = std::min(x + 1, grid_size_x - 1);
		const uint y_begin = std::max(y - 1, 0u);
		const uint y_end = std::min(y + 1, grid_size_y - 1);
		const uint z_begin = std::max(z - 1, 0u);
		const uint z_end = std::min(z + 1, grid_size_z - 1);
	}
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
		boids_grid_id = new uint[N]();
	}
	~Flock() {
		delete[] boids_p;
		delete[] boids_v;
		delete[] boids_grid_id;
	}
	void update(float dt) {
		updateGrid();
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
				(center - boids_p[i]) * centeringFactor
				+ close * avoidFactor
				+ (vel - boids_v[i]) * matchingFactor;



			boundPosition(i);
			boundVelocity(i);
			boids_p[i] += boids_v[i] * dt;
		}
	}
	void boundVelocity(int i) {
		float speed = glm::l1Norm(boids_v[i]);
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
		return close * avoidFactor;
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

