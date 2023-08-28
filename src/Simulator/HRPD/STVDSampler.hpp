#pragma once

#include <queue>
#include <set>
#include <string>
#include <vector>

#include "Simulator/PDTypeDef.hpp"
#include "spdlog/spdlog.h"

#define STVD_MODE_GRAPH_DIJKSTRA 0
#define STVD_MODE_ST_DIJKSTRA 1
#define STVD_DEFAULT_K 10

namespace PD {

struct STVDSampler {
    unsigned int m_numVerts;
    std::vector<unsigned int> m_srcVerts;
    std::vector<std::vector<unsigned int>> m_adjVerts;
    PDVector m_dists;
    PDPositions m_positions;
    unsigned int m_lastVert;

    // pre-step for dijkstra distance computation
    unsigned int m_k{ 10 };

    struct QueVert {
        unsigned int vert;
        double dist;
        QueVert(unsigned int v, double d)
            : vert(v), dist(d) {}
    };

    class QueVertComp {
    public:
        bool operator()(QueVert v1, QueVert v2) { return v1.dist > v2.dist; }
    };

public:
    double updateVertDist(unsigned int v1, unsigned int v2, std::vector<int>& predecessor, unsigned int mode)
    {
        double bestDist = -1;
        if (mode == STVD_MODE_GRAPH_DIJKSTRA) {
            bestDist = m_dists(v1) + (m_positions.row(v1) - m_positions.row(v2)).norm();
        }
        else if (mode == STVD_MODE_ST_DIJKSTRA) {
            // ? Implementation of the STVD update_dist function from [Campen et al. 2013]
            int curV = v1;
            int nxtV = v2;
            int preV = predecessor[curV];
            PD3dVector curEdge = m_positions.row(nxtV) - m_positions.row(curV);
            bestDist = m_dists[v1] + curEdge.norm();
            for (int i = 1; i < m_k; i++) {
                preV = predecessor[curV];
                if (preV < 0)
                    break;

                curEdge = m_positions.row(nxtV) - m_positions.row(preV);
                double curDist = m_dists[preV] + curEdge.norm();
                if (curDist < bestDist)
                    bestDist = curDist;

                curV = preV;
            }
        }
        return bestDist;
    }

    void computeDistances(bool update = false, unsigned int mode = STVD_MODE_GRAPH_DIJKSTRA, PDScalar maxDist = -1)
    {
        if (!update) {
            m_dists.setConstant(-1.);
        }
        std::priority_queue<QueVert, std::vector<QueVert>, QueVertComp> queue;
        std::vector<int> predecessor(m_numVerts, -1);
        std::vector<bool> isFinal(m_numVerts, false);

        for (auto v : m_srcVerts) {
            m_dists[v] = 0;
            queue.push(QueVert(v, 0));
        }

        int loopCnt = 0;
        m_lastVert = -1;
        while (!queue.empty()) {
            QueVert qv = queue.top();
            queue.pop();
            if (isFinal[qv.vert])
                continue;
            loopCnt++;
            m_lastVert = qv.vert;
            isFinal[qv.vert] = true;
            if (maxDist > 0 && m_dists[qv.vert] > maxDist)
                continue;
            // spdlog::info(">>> qv = {}, adj size = {}", qv.vert, m_adjVerts[qv.vert].size());
            for (auto nv : m_adjVerts[qv.vert]) {
                if (isFinal[nv])
                    continue;
                double updateDist = updateVertDist(qv.vert, nv, predecessor, mode);
                if (m_dists[nv] < 0 || updateDist < m_dists[nv]) {
                    m_dists[nv] = updateDist;
                    predecessor[nv] = qv.vert;
                    queue.push(QueVert(nv, m_dists[nv]));
                }
            }
        }
    }

public:
    void Init(PDPositions const& V, PDTriangles const& F, PDTets const& Tets)
    {
        m_numVerts = V.rows();
        m_srcVerts.clear();
        m_adjVerts.resize(m_numVerts);
        m_dists.resize(m_numVerts);
        m_dists.setConstant(-1.);
        m_positions = V.cast<PDScalar>();

        // Establish neighbourhood structure for tets or triangles
        for (unsigned int tet = 0; tet < Tets.rows(); tet++) {
            for (unsigned int locV1 = 0; locV1 < 4; locV1++) {
                int locV1Ind = Tets(tet, locV1);
                for (unsigned int locV2 = 0; locV2 < 4; locV2++) {
                    if (locV2 != locV1) {
                        int locV2Ind = Tets(tet, locV2);
                        if (std::find(m_adjVerts.at(locV1Ind).begin(), m_adjVerts.at(locV1Ind).end(), locV2Ind) == m_adjVerts.at(locV1Ind).end()) {
                            m_adjVerts.at(locV1Ind).push_back(locV2Ind);
                        }
                    }
                }
            }
        }
    }

    double GetMaxDist(std::vector<unsigned int>& samples)
    {
        m_srcVerts = samples;
        computeDistances();
        double furthestDist = std::max(0., m_dists.maxCoeff());
        return furthestDist;
    };

    void ExtendSamples(unsigned int numSamples, std::vector<unsigned int>& samples)
    {
        // O(numSamples * N * log(N))
        for (int i = 0; i < numSamples; i++) {
            computeDistances(true, STVD_MODE_GRAPH_DIJKSTRA);
            int vert = -1;
            PDScalar maxDist = -1;
            for (int v = 0; v < m_dists.size(); v++) {
                if (maxDist < m_dists[v]) {
                    vert = v;
                    maxDist = m_dists[v];
                }
            }
            if (vert == -1) {
                spdlog::error("Error during sampling, returning with fewer m_samples... ");
                break;
            }
            else if (std::find(samples.begin(), samples.end(), vert) != samples.end()) {
                spdlog::error("i = {}, vert = {}", i, vert);
                spdlog::error("Duplicate vertex was selected in sampling, canceling.");
                break;
            }
            m_srcVerts.push_back(vert);
            samples.push_back(vert);
        }
    }

    std::vector<unsigned int> GetSamples(unsigned int numSamples)
    {
        std::vector<unsigned int> _samples;
        m_srcVerts.clear();
        m_dists.setConstant(-1.);

        // Add first sample
        unsigned int firstVert = 667; // CTODO
        // unsigned int firstVert = rand() % (unsigned int)m_restpose_positions.rows();
        _samples.push_back(firstVert);
        m_srcVerts.push_back(firstVert);

        ExtendSamples(numSamples - 1, _samples);

        std::sort(_samples.begin(), _samples.end());
        return _samples;
    }
};
} // namespace PD
