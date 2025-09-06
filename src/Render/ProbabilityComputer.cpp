#pragma once
#include "ProbabilityComputer.hpp"
#include <functional>
#include "../../external/opencv/build/include/opencv2/core/mat.hpp"


static unsigned char clampToUchar(double x, double min, double max)
{
	if (x < min) { return static_cast<unsigned char>(min); }
	if (x > max) { return static_cast<unsigned char>(max); }
	return static_cast<unsigned char>(x);
}

void ProbabilityComputer::fillImage(cv::Mat* image, const std::vector<Star>& starVec, const double& posMax2)
{
	double totalMass{ 0. };
	std::vector<double> coefMatrix( m_imgSize * m_imgSize, 0. );
	float rap = static_cast<float>(m_imgSize) / static_cast<float>(posMax2);
	for (const Star& star : starVec)
	{
		auto newPos = star.position * rap;
		size_t idx = getMatrixCoord(newPos, posMax2);
		if (idx > 0)
		{
			coefMatrix[idx] += static_cast<double>(star.mass);
		}
		totalMass += star.mass;
	}
	if (m_maxValue < 0)
	{
		m_maxValue = *std::max_element(coefMatrix.begin(), coefMatrix.end());
	}
	if (m_maxValue > 0)
	{
		std::transform(coefMatrix.begin(), coefMatrix.end(), coefMatrix.begin(),
			std::bind(std::multiplies<double>(), std::placeholders::_1, 1. / m_maxValue));
	}


	for (size_t row{ 0u }; row < m_imgSize; ++row)
	{
		for (size_t col{ 0u }; col < m_imgSize; ++col)
		{
			Pixel p = getPixel(coefMatrix.at(row + col * m_imgSize));
			cv::Vec3b pixel;
			pixel[2] = p.r;
			pixel[1] = p.g;
			pixel[0] = p.b;
			image->at<cv::Vec3b>({ static_cast<int>(row), static_cast<int>(col) }) = pixel;
		}
	}
}

size_t ProbabilityComputer::getMatrixCoord(const PosLy& s, const double& posMax2) const
{
	size_t x = size_t(s.x) + m_imgSize / 2;
	size_t y = size_t(s.y) + m_imgSize / 2;
	if (
		x < m_imgSize && x >= 0 &&
		y < m_imgSize && y >= 0
		)
	{
		return x + y * m_imgSize;
	}
	return 0;
}

Pixel ProbabilityComputer::getPixel(double coef) const
{
	if (coef < 1e-5) return Pixel(0u, 0u, 0u);
	double newCoef = std::log(1 + 20 * coef) / std::log(20);

	return Pixel{ 
		clampToUchar(newCoef * 235., 15, 255),
		clampToUchar(newCoef * 235., 15, 255), 
		clampToUchar(newCoef * 255., 15, 255) };
}
