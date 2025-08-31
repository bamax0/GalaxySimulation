#pragma once
#include "TreeNode.hpp"
#include <concurrent_queue.h>
#include <thread>

template<unsigned char N>
class GALAXY_SIM_DLL_EXPORT FirstTreeNode : public ITreeNode
{
	struct ThreadData
	{
		ITreeNode* node{ nullptr };
		Concurrency::concurrent_queue<std::reference_wrapper<const Star>> starQueue{};
		bool isFinished = false;
	};
public:
	FirstTreeNode(SimplifiedTreeNode<N>* treeNode) : ITreeNode(), m_rootNode(treeNode) {}
	~FirstTreeNode() { delete m_rootNode; }

	FirstTreeNode(const FirstTreeNode&) = delete;
	FirstTreeNode& operator=(const FirstTreeNode&) = delete;

	void reset(const Bbox& bbox) override
	{
		m_rootNode->reset(bbox);
	}
	void startInserting() override;
	void endInserting() override;
	void appendStar(const Star& star, size_t depth = 0) override;

	Vec3 computeTotalForce(const Star& star) const override
	{
		return m_rootNode->computeTotalForce(star);
	}

private:
	static void threadFunc(ThreadData* data);
	

	SimplifiedTreeNode<N>* m_rootNode;
	std::vector<ThreadData> m_dataQueus;
	std::vector<std::thread> m_threadList;
};

template<unsigned char N>
void FirstTreeNode<N>::threadFunc(ThreadData* data)
{
	Star empty;
	std::reference_wrapper<const Star> s{ empty };
	while (!data->starQueue.empty() || !data->isFinished)
	{
		bool isOk = data->starQueue.try_pop(s);
		if (isOk)
		{
			data->node->appendStar(s.get());
		}
	}
	return;
}

template<unsigned char N>
void FirstTreeNode<N>::startInserting()
{
	m_rootNode->createAllChilds();

	m_threadList.clear();
	m_dataQueus.clear();

	m_threadList.reserve(N);
	m_dataQueus.resize(N);
	for (unsigned char i = 0u; i < N; ++i)
	{
		m_dataQueus[i].node = m_rootNode->getChild(i);
		m_dataQueus[i].starQueue.clear();
		m_dataQueus[i].isFinished = false;

		m_threadList.emplace_back(
			std::thread(FirstTreeNode<N>::threadFunc, 
				&m_dataQueus[i])
		);
	}
}

template<unsigned char N>
void FirstTreeNode<N>::endInserting()
{
	for (ThreadData& dataQueu : m_dataQueus)
	{
		dataQueu.isFinished = true;
	}
	for (std::thread& thread : m_threadList)
	{
		thread.join();
	}
	m_threadList.clear();
	m_dataQueus.clear();

	// m_rootNode->m_bbox 
}

template<unsigned char N>
void FirstTreeNode<N>::appendStar(const Star& star, size_t depth)
{
    size_t idx = m_rootNode->getIndex(star.position);
	m_dataQueus[idx].starQueue.push({ star });
}