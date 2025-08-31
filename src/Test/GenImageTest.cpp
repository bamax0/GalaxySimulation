#include <gtest/gtest.h>
#include <Physics/Star.hpp>
#include <chrono>
#include "Render/ImageGen.hpp"

TEST(ImageTest, testGenImage)
{
	auto projectDir = std::filesystem::absolute( std::filesystem::path(u8"..") / u8".." / u8".." / u8"..");
	// EXPECT_EQ(projectDir, "");
	ImageGen gen;
	gen.init(projectDir / u8"test_data" / u8"output");


	std::vector<Star> stars;
	for (float i = 0; i < 1000; i++)
	{
		stars.push_back({});
		stars.back().position = Vec3{ 2.f * i, i, 3.f * i };
	}

	gen.generateImg(stars, {0, 0, 0}, 3000, 0u);

	for (float i = 0; i < 1000; i++)
	{
		stars.push_back({});
		stars.back().position = Vec3{ 2.f * i, 3.f*i, i};
	}
	gen.generateImg(stars, { 0, 0, 0 }, 3000, 1u);
}
