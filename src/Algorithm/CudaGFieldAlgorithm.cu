#include "Algorithm/CudaGFieldAlgorithm.hpp"
#include "Physics/function.hpp"
#include <cuda_runtime.h>
#include <iostream>


struct GFieldNode
{
	Vec3 m_center;
	Vec3 field;
};

__device__ __host__ __inline__ Vec3 interpolate(const GFieldNode& c000, const GFieldNode& c001, 
	const GFieldNode& c010, const GFieldNode& c011, const GFieldNode& c100, const GFieldNode& c101, 
	const GFieldNode& c110, const GFieldNode& c111, const Vec3 pos)
{
	// TODO
	float x_min = std::floor(pos.x);
	float y_min = std::floor(pos.y);
	float z_min = std::floor(pos.z);

	float x_max = std::floor(pos.x);
	float y_max = std::floor(pos.y);
	float z_max = std::floor(pos.z);

	float xd = (v.x - x_min) / (x_max - x_min);
	float yd = (v.y - y_min) / (x_max - y_min);
	float zd = (v.z - z_min) / (x_max - z_min);


	Vec3 c00 = c000.field * (1 - xd) + c100.field * xd;
	Vec3 c01 = c001.field * (1 - xd) + c101.field * xd;
	Vec3 c10 = c010.field * (1 - xd) + c110.field * xd;
	Vec3 c11 = c011.field * (1 - xd) + c111.field * xd;

	Vec3 c0 = c00 * (1 - yd) + c10 * yd;
	Vec3 c1 = c01 * (1 - yd) + c11 * yd;

	return c0 * (1 - zd) + c1 * zd;
}

__global__ void computeForces(Star* stars, size_t nbStars, GFieldNode* gfield, size_t fieldSize, TimeS dt)
{
	size_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= nbStars) return;
	Vec3 f{ 0.f, 0.f, 0.f };
	stars[id];


	leapFrog(stars[id], getAcceleration(stars[id], f), dt);
	return;
}

__global__ void computeGField(Star* stars, size_t nbStars, GFieldNode* gfield, size_t fieldSize)
{
	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	size_t idy = threadIdx.y + blockIdx.y * blockDim.y;
	size_t idz = threadIdx.z + blockIdx.z * blockDim.z;
	if (idx >= fieldSize) return;
	if (idy >= fieldSize) return;
	if (idz >= fieldSize) return;
	Star center;
	center.mass = 1.f; // Avec une mass = 1, ca équivaut a calculé le champ

	center.position = gfield[idx + idy*fieldSize +  idz*fieldSize* fieldSize].m_center;

	Vec3 f{ 0.f, 0.f, 0.f };
	for (size_t i{ 0u }; i < nbStars; ++i)
	{
			f += computeForce(center, stars[i]);
	}
	gfield[idx + idy * fieldSize].field = f;
	return;
}


namespace GalaxySim
{
	CudaGFieldAlgorithm::initWithGPU(size_t nbStars, Star* stars)
	{
		m_nbStars = m_nbStars;
		m_gpuStarsPtr = stars;


	}

	CudaGFieldAlgorithm::~CudaGFieldAlgorithm()
	{
		// todo
	}

	void CudaGFieldAlgorithminit(const std::vector<Star>& stars) {/* Not inplemented*/ }


	void CudaGFieldAlgorithm::next(TimeY dt) override
	{
	}

	virtual std::vector<Star> CudaGFieldAlgorithm::getStars()
	{
	}

	const Star* CudaGFieldAlgorithm::getCudaStars() const
	{
	}

		// Star* m_gpuStarsPtr{ nullptr };
		// Vec3* m_gpuFieldPtr{ nullptr };
		// size_t m_nbStars{ 0u };
		// size_t m_fieldSize{ 0u };
}