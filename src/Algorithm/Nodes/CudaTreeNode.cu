#pragma once
#include "CudaTreeNode.hpp"

OctreePtr CudaTreeNode::createCudaPtr() const
{
	CudaTreeNodeFlat* nodesCuda;
	size_t sizeOfMem = m_nodes->size() * sizeof(CudaTreeNodeFlat);
	//std::cout << "Cb nodes de l'octree cuda: " << m_nodes->size() << std::endl;
	cudaMalloc((void**)&nodesCuda, sizeOfMem);
	cudaMemcpy(nodesCuda, m_nodes->data(), sizeOfMem, cudaMemcpyHostToDevice);

	size_t sizeOfMemLastNode = m_lastNodes->size() * sizeof(CudaLastTreeNodeFlat);
	CudaLastTreeNodeFlat* lastNodesCuda;
	//std::cout << "Cb dernier  nodes de l'octree cuda: " << m_lastNodes->size() << std::endl;
	cudaMalloc((void**)&lastNodesCuda, sizeOfMemLastNode);
	cudaMemcpy(lastNodesCuda, m_lastNodes->data(), sizeOfMemLastNode, cudaMemcpyHostToDevice);
	return OctreePtr{nodesCuda, lastNodesCuda };
}

void CudaTreeNode::reset(const Bbox& bbox)
{
	SimplifiedTreeNode<8>::reset(bbox);
	m_nodeIdx =  0u;
	m_nodes->clear();
	m_nodes->push_back(
		CudaTreeNodeFlat{ PosLy(), 0.f, m_bbox.size * 2.f / s_theta }
	);
}

unsigned char CudaTreeNode::getIndex(const PosLy& pos) const
{
	unsigned char idx{ 0u };

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

Bbox CudaTreeNode::getSubBbox(unsigned char id) const
{
	return Bbox{
		{m_bbox.center.x + m_bbox.size * ((id & 1) ? 0.5f : -0.5f),
		 m_bbox.center.y + m_bbox.size * ((id & 2) ? 0.5f : -0.5f),
		 m_bbox.center.z + m_bbox.size * ((id & 4) ? 0.5f : -0.5f) },
	m_bbox.size / 2.f
	};
}

SimplifiedTreeNode<8>* CudaTreeNode::createChild(const Bbox& bbox, unsigned char idx) const
{
	CudaTreeNodeFlat nodeCuda{};
	nodeCuda.barnesHutCst = bbox.size * 2.f / s_theta;
	nodeCuda.centerOfMass = Vec3();
	nodeCuda.mass = 0.f;

	std::lock_guard<std::mutex> lock(*m_mutex.get());
	size_t childIdx = m_nodes->size();
	m_nodes->push_back(nodeCuda);
	m_nodes->at(m_nodeIdx).setChildIdx(idx, static_cast<int64_t>(childIdx));
	return new CudaTreeNode(m_mutex, m_nodes, m_lastNodes, childIdx, bbox);
}

ITreeNode* CudaTreeNode::createLastTreeNode(const PosLy& pos, MassMs mass, unsigned char idx) const
{
	CudaLastTreeNode* lastNode = new CudaLastTreeNode(m_mutex, m_lastNodes, pos, mass);
	int64_t childIdx = -static_cast<int64_t>(lastNode->getIndex()) - 1;

	std::lock_guard<std::mutex> lock(*m_mutex.get());
	m_nodes->at(m_nodeIdx).setChildIdx(idx, childIdx);
	return lastNode;
}