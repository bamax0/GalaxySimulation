#include <gtest/gtest.h>
#include <Physics/Star.hpp>
#include <GalaxyGenerator/GalaxyGenerator.hpp>
#define private public
#define protected public
#include <Algorithm/BarnesHutAlgorithm.hpp>
#include <Algorithm/Nodes/OctreeNode.hpp>
#include <Algorithm/Nodes/SedecTreeNode.hpp>
#include <Algorithm/Nodes/FirstTreeNode.hpp>
#undef private
#undef protected

#define EXPECT_BBOX_EQ(bbox0, bbox1)  \
{ \
EXPECT_FLOAT_EQ((bbox0).center.x, (bbox1).center.x); \
EXPECT_FLOAT_EQ((bbox0).center.y, (bbox1).center.y); \
EXPECT_FLOAT_EQ((bbox0).center.z, (bbox1).center.z); \
EXPECT_FLOAT_EQ((bbox0).size, (bbox1).size); \
}

TEST(NodeTest, OctreeBbox)
{
	OctreeNode node{ Bbox{{0.f, 0.f, 0.f}, 4.f} };
	{
		size_t idx = node.getIndex({ -3, -3, -3 });
		Bbox bb{ {-2.f, -2.f, -2.f}, 2.f };
		EXPECT_BBOX_EQ(node.getSubBbox(idx), bb);
	}
	{
		size_t idx = node.getIndex({ 3, 3, 3 });
		Bbox bb{ {2.f, 2.f, 2.f}, 2.f };
		EXPECT_BBOX_EQ(node.getSubBbox(idx), bb);
	}
	{
		size_t idx = node.getIndex({ 0.5, -1, 3 });
		Bbox bb{ {2.f, -2.f, 2.f}, 2.f };
		EXPECT_BBOX_EQ(node.getSubBbox(idx), bb);
	}
}

TEST(NodeTest, SedecTreeBbox)
{
	SedecTreeNode node{ Bbox{{0.f, 0.f, 0.f}, 4.f} };
	{
		size_t idx = node.getIndex({ -3, -3, -3 });
		Bbox bb{ {-3.f, -3.f, -3.f}, 1.f };
		EXPECT_BBOX_EQ(node.getSubBbox(idx), bb);
	}
	{
		size_t idx = node.getIndex({ 3, 3, 3 });
		Bbox bb{ {3.f, 3.f, 3.f}, 1.f };
		EXPECT_BBOX_EQ(node.getSubBbox(idx), bb);
	}
	{
		size_t idx = node.getIndex({ 0.5, -1, 3 });
		Bbox bb{ {1.f, -1.f, 3.f}, 1.f };
		EXPECT_BBOX_EQ(node.getSubBbox(idx), bb);
	}
}
	