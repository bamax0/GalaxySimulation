#pragma once
#include "ITreeNode.hpp"
#include "Physics/function.hpp"
#include <iostream>
#include <numeric>
#include <vector>

class GALAXY_SIM_DLL_EXPORT LastTreeNode :public ITreeNode
{
public:
	LastTreeNode(const Star& star) : m_stars()
	{
		m_stars.push_back(star);
	}
	virtual ~LastTreeNode() override = default;
	void reset(const Bbox& bbox) override { m_stars.clear(); };
	void startInserting() override {};
	void endInserting() override {};
	void appendStar(const Star& star, size_t depth = 0) override
	{
		m_stars.push_back(star);
	}
	Vec3 computeTotalForce(const Star& star) const override
	{	
		return {};
		/*
		auto addForce = [&star](Vec3 a, Star s)
			{
				return std::move(a) + computeForce(star, s);
			};
		Vec3 force{ std::accumulate(m_stars.begin(), m_stars.end(), Vec3(0.f, 0.f, 0.f), addForce) };
		// for (const Star& s : m_stars)
		// {
		// 	force += computeForce(star, s);
		// }
		return force;
		*/
	}
private:
	std::vector<Star> m_stars;
};