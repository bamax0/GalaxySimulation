#pragma once
#include "TreeNode.hpp"

class GALAXY_SIM_DLL_EXPORT OctreeNode : public TreeNode<8>
{
public:
	OctreeNode(const Bbox& bbox) : TreeNode<8>(bbox) {}
	OctreeNode(const Bbox& bbox, const PosLy& pos, MassMs mass) : TreeNode<8>(bbox, pos, mass) {}
protected:

	unsigned char getIndex(const PosLy& pos) const override
	{
		size_t idx{ 0u };

		if (pos.x > m_bbox.center.x)
		{
			idx |= 1;
		}
		
		if (pos.y > m_bbox.center.y)
		{
			idx |= 2;
		}
		
		if (pos.z > m_bbox.center.z)
		{
			idx |= 4;
		}
		return idx;
	}

	Bbox getSubBbox(unsigned char id) const override
	{
		return Bbox{
			{m_bbox.center.x + m_bbox.size * ((id & 1) ? 0.5f : -0.5f),
			 m_bbox.center.y + m_bbox.size * ((id & 2) ? 0.5f : -0.5f),
			 m_bbox.center.z + m_bbox.size * ((id & 4) ? 0.5f : -0.5f) },
		m_bbox.size / 2.f
		};
	}

	TreeNode<8>* createChild(const Bbox& bbox, unsigned char) const override
	{
		return new OctreeNode(bbox);
	}
	
	using TreeNode<8>::update;
	using TreeNode<8>::getCenterOfMass;
	using TreeNode<8>::getMass;

	using TreeNode<8>::reset;
	using TreeNode<8>::startInserting;
	using TreeNode<8>::endInserting;
	using TreeNode<8>::appendStar;
	// using TreeNode<8>::computeTotalForce;
	using TreeNode<8>::createLastTreeNode;
	using TreeNode<8>::createChild;
};