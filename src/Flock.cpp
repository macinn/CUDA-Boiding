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
	
	float grid_size;
	int grid_size_x;
	int grid_size_y;
	int grid_size_z;

	uint getGridId(uint x, uint y, uint z) {
		return x + y * grid_size_x + z * grid_size_x * grid_size_y;
	}
	uint getGridId(glm::vec3 pos) {
		return getGridId(pos.x / grid_size, pos.y / grid_size, pos.z / grid_size);
	}
	void getFromGridId(uint index, int& x, int& y, int& z) {
		if (index == 0) {
			x = 0;
			y = 0;
			z = 0;
		}
		else {
			x = index % grid_size_x;
			y = (index / grid_size_x) % grid_size_y;
			z = index / (grid_size_x * grid_size_y);
		}
	}
	void updateGrid() {
		for (uint i = 0; i < N; i++) {
			boids_grid_id[i] = getGridId(boids_p[i]);
		}	
	}
	void init() {
		std::default_random_engine rd{ static_cast<long unsigned int>(time(0)) };
		std::mt19937 gen{ rd() };
		std::uniform_real_distribution<> w(0, width);
		std::uniform_real_distribution<> h(0, height);
		std::uniform_real_distribution<> z(0, depth);

		for (uint i = 0; i < N; i++) {
			boids_p[i] = glm::vec3(w(gen), h(gen), z(gen));
		}
	}
public:
	// Parameters
	float turnFactor = 0.2f;
	float visualRange = 5.f;	
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
		setVisualRange(visualRange);
		init();
	}
	void setVisualRange(float visualRange) {
		this->visualRange = visualRange;
		this->grid_size = 1.f * visualRange;
		this->grid_size_x = (width - 1) / grid_size + 1;
		this->grid_size_y = (height - 1) / grid_size + 1;
		this->grid_size_z = (depth - 1) / grid_size + 1;
	}
	~Flock() {
		delete[] boids_p;
		delete[] boids_v;
		delete[] boids_grid_id;
	}
	void update(float dt) {

		updateGrid();
		const float visualRangeSquared = visualRange * visualRange;
		const float protectedRangeSquared = protectedRange * protectedRange;

		for (uint i = 0; i < N; i++) {

			uint countVisible = 0;
			uint countClose = 0;
			glm::vec3 vel = glm::vec3(0.0f);
			glm::vec3 center = glm::vec3(0.0f);
			glm::vec3 close = glm::vec3(0.0f);
			const uint index_i = boids_grid_id[i];
			int x_i, y_i, z_i;
			getFromGridId(index_i, x_i, y_i, z_i);
			const int x_begin = std::max(x_i - 1, 0);
			const int x_end = std::min(x_i + 1, grid_size_x - 1);
			const int y_begin = std::max(y_i - 1, 0);
			const int y_end = std::min(y_i + 1, grid_size_y - 1);
			const int z_begin = std::max(z_i - 1, 0);
			const int z_end = std::min(z_i + 1, grid_size_z - 1);

			for (uint j = 0; j < N; j++) {
				int x_j, y_j, z_j;
				getFromGridId(boids_grid_id[j], x_j, y_j, z_j);
				if (x_j >= x_begin && x_j <= x_end
					&& y_j >= y_begin && y_j <= y_end
					&& z_j >= z_begin && z_j <= z_end
					&& i != j) {
					const float distanceSquared = glm::distance2(boids_p[i], boids_p[j]);
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

