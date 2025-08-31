#include "Algorithm/CpuSimpleAlgorithm.hpp"
#include "Physics/function.hpp"
#include <iostream>
#include <execution>

namespace GalaxySim
{
	void CpuSimpleAlgorithm::init(const std::vector<Star>& stars)
	{
		m_nbStars = stars.size();
		m_stars = stars;
	}

	void CpuSimpleAlgorithm::next(TimeY dt)
	{
		std::for_each(
			std::execution::par,
			m_stars.begin(),
			m_stars.end(),
			[this, dt](Star& star)
			{
				Vec3 force{ 0.f, 0.f, 0.f };
				for (size_t i{ 0u }; i < m_nbStars; ++i)
				{
					force += computeForce(star, m_stars[i]);
				}
				Vec3 acc = getAcceleration(star, force);
				leapFrog(star, acc, dt);
			});
	}

	std::vector<Star> CpuSimpleAlgorithm::getStars() const
	{
		return m_stars;
	}
}