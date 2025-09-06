#pragma once
#include "ITreeNode.hpp"
#include "Physics/function.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

/* Algorithme inéfficace */
class GALAXY_SIM_DLL_EXPORT LastTreeNodeUsingCuda :public ITreeNode 
{
public:
	LastTreeNodeUsingCuda(const Star& star) : m_stars()
	{
		m_stars.push_back(star);
	}
	
	virtual ~LastTreeNodeUsingCuda() override
	{
		reset({});
	}

	void reset(const Bbox& bbox) override;
	void startInserting() override {};
	void endInserting() override {};
	void appendStar(const Star& star, size_t depth) override
	{
		++m_nbStars;
		m_stars.push_back(star);
	}

	Vec3 computeTotalForce(const Star& star) const override;

	Vec3 computeTotalForceWithoutCuda(const Star& star) const
	{
		Vec3 force{ 0.f, 0.f, 0.f };
		for (const Star& s : m_stars)
		{
			force += computeForce(star, s);
		}
		return force;
	}
private:
	void initCUda() const;

	std::vector<Star> m_stars;
	mutable Star* m_gpuStarsPtr{ nullptr };
	mutable Vec3* m_forces{ nullptr };
	mutable size_t m_nbStars{ 0u };
};