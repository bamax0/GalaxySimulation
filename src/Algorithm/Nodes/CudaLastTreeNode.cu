#include "CudaLastTreeNode.hpp"

CudaLastTreeNode::CudaLastTreeNode(
	std::shared_ptr<std::mutex> mutex,
	std::shared_ptr<std::vector<CudaLastTreeNodeFlat>> nodes,
	const Vec3& starPos, MassMs mass): m_nodes(nodes), m_mutex(mutex)
{
	std::lock_guard<std::mutex> lock(*m_mutex.get());
	m_nodeLastIdx = m_nodes->size();
	m_nodeFirstIdx = m_nodeLastIdx;
	m_nodes->push_back(CudaLastTreeNodeFlat{ starPos, mass });
}


void CudaLastTreeNode::appendStar(const Star& star, size_t depth)
{
	std::lock_guard<std::mutex> lock(*m_mutex.get());
	CudaLastTreeNodeFlat& node = m_nodes->at(m_nodeLastIdx);
	m_nodeLastIdx = m_nodes->size();
	node.childIdx = m_nodeLastIdx;
	//printf("add: last %i first %i\n", m_nodeLastIdx, m_nodeFirstIdx);
	m_nodes->push_back(CudaLastTreeNodeFlat{ star.position, star.mass });
}