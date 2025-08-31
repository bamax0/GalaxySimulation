#include <gtest/gtest.h>
#include <Physics/Star.hpp>
#include <Algorithm/CudaSimpleAlgorithm.hpp>
#include <Algorithm/CpuSimpleAlgorithm.hpp>
#include <GalaxyGenerator/GalaxyGenerator.hpp>
#include <Utils/Chrono.hpp>

TEST(Validity, testGPU)
{
	GalaxySim::CudaSimpleAlgorithm algGPU{};

	GenParam param{};
	param.min = -1.;
	param.min = 1.;
	param.maxMass = 100000.;
	param.minMass = 0.1;
	param.maxSpeed = 0.1;
	param.minSpeed = 0;

	std::cout << "init" << std::endl;
	TestChrono::start();
	auto stars = getSquareUniversFromGPUToCPU(static_cast<size_t>(1e10), param);
	TestChrono::end();
	TestChrono::show();

	algGPU.init(stars);

	for (int step{ 0 }; step < 10; ++step)
	{
		std::cout << "gpu" << std::endl;
		TestChrono::start();
		algGPU.next(10000.);
		TestChrono::end();
		TestChrono::show();
	}
}

TEST(Validity, compareCPU_GPU)
{
	GalaxySim::CudaSimpleAlgorithm algGPU{};
	GalaxySim::CpuSimpleAlgorithm algCPU{};

	GenParam param{};
	param.min = -1.;
	param.min = 1.;
	param.maxMass = 100000.;
	param.minMass = 0.1;
	param.maxSpeed= 0.1;
	param.minSpeed= 0;

	std::cout << "init" << std::endl;
	TestChrono::start();
	auto stars = getSquareUniversFromGPUToCPU(static_cast<size_t>(1e5), param);
	TestChrono::end();
	TestChrono::show();
	algCPU.init(stars);
	algGPU.init(stars);

	for (int step{ 0 }; step < 15; ++step)
	{
		std::cout << "cpu" << std::endl;
		TestChrono::start();
		algCPU.next(1000.);
		TestChrono::end();
		TestChrono::show();

		std::cout << "gpu" << std::endl;
		TestChrono::start();
		algGPU.next(1000.);
		TestChrono::end();
		TestChrono::show();

		auto str2 = algGPU.getStars();
		for (int i = 0; i < stars.size(); ++i)
		{
			EXPECT_NEAR(stars.at(i).position.x, str2.at(i).position.x, 1e-20);
			EXPECT_NEAR(stars.at(i).position.y, str2.at(i).position.y, 1e-20);
			EXPECT_NEAR(stars.at(i).position.z, str2.at(i).position.z, 1e-20);
		}
	}
}


Vec3 f(Vec3 v)
{
	float a = Vec3::dot(v, v);
	return (std::exp(0.01/(a+.1)) - a * 0.001)* v + Vec3(1.f, 5.f, 4.f);
}


Vec3 interpolate(Vec3 v)
{
	float x = std::floor( v.x );
	float y = std::floor( v.y );
	float z = std::floor( v.z );

	Vec3 c000(f(Vec3{ x, y, z }));
	Vec3 c001(f(Vec3{ x, y, z + 1.f }));
	Vec3 c010(f(Vec3{ x, y + 1.f, z }));
	Vec3 c011(f(Vec3{ x, y + 1.f, z + 1 }));
	Vec3 c100(f(Vec3{ x + 1, y, z }));
	Vec3 c101(f(Vec3{ x + 1, y, z + 1 }));
	Vec3 c110(f(Vec3{ x + 1, y + 1, z }));
	Vec3 c111(f(Vec3{ x + 1, y + 1, z + 1 }));

	float xd = (v.x - x) / 1;
	float yd = (v.y - y) / 1;
	float zd = (v.z - z) / 1;


	Vec3 c00 = c000 * (1 - xd) + c100 * xd;
	Vec3 c01 = c001 * (1 - xd) + c101 * xd;
	Vec3 c10 = c010 * (1 - xd) + c110 * xd;
	Vec3 c11 = c011 * (1 - xd) + c111 * xd;

	Vec3 c0 = c00 * (1 - yd) + c10 * yd;
	Vec3 c1 = c01 * (1 - yd) + c11 * yd;

	return c0 * (1 - zd) + c1 * zd;
}

#define EXPECT_EQ_VEC(a, b) EXPECT_NEAR(a.x, b.x, 1e-1f);EXPECT_NEAR(a.y, b.y, 1e-1f);EXPECT_NEAR(a.z, b.z, 1e-1f);

TEST(Interpolate, test)
{
	Vec3 v({0.5741, 0.11477, -0.174});
	for (float k = -10; k < 10; k += 0.01457)
	{
		EXPECT_EQ_VEC(interpolate(k*v), f(k*v));
	}
}

TEST(Optimisation, SIMD)
{
	Vec3 a{ 1, 2, 3 };
	Vec3 b{ 5, 8, 1 };
	TestChrono::start();
	double nbMax = 1e8;
	for (double i{ 0 }; i < nbMax; ++i)
	{
		a.x= std::min(a.x, b.x);
		a.y= std::min(a.y, b.y);
		a.z= std::min(a.z, b.z);
	}
	EXPECT_FLOAT_EQ(a.x, 1);
	TestChrono::end();
	TestChrono::show();

	a = { 1, 2, 3 };
	__m128 va = _mm_loadu_ps(&a.x);
	TestChrono::start();
	for (double i{ 0 }; i < nbMax; ++i)
	{
		__m128 vb = _mm_loadu_ps(&b.x);
		va = _mm_min_ps(va, vb);
	}
	_mm_storeu_ps(&a.x, va);
	EXPECT_FLOAT_EQ(a.x, 1);
	EXPECT_FLOAT_EQ(a.y, 2);
	EXPECT_FLOAT_EQ(a.z, 3);
	TestChrono::end();
	TestChrono::show();

	EXPECT_TRUE(false);
}