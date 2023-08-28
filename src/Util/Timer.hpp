#pragma once

#include <chrono>
#include <string>
#include <map>
#include <spdlog/spdlog.h>
using namespace std::chrono;

#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK(x) spdlog::warn("{}: {} s", #x, std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench_##x).count());
#define TICKC(x) auto cbench_##x = std::chrono::steady_clock::now();
#define TOCKC(x) spdlog::critical("{}: {} s", #x, std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - cbench_##x).count());

namespace Util {
struct TimePair {
    high_resolution_clock::time_point tStart, tEnd;
};

class StopWatch {
private:
    std::vector<TimePair> m_measurements;
    bool m_running;
    high_resolution_clock::time_point m_curTStart;
    unsigned int m_predictedMeasurements, m_maxMeasurements;

public:
    StopWatch(unsigned int predictedMeasurements = 10000, unsigned int maxMeasurements = 100000)
        : m_predictedMeasurements(predictedMeasurements), m_maxMeasurements(maxMeasurements), m_curTStart(), m_running(false), m_measurements(0)
    {
        m_measurements.reserve(predictedMeasurements);
    }

    inline void reset()
    {
        m_measurements.clear();
        m_measurements.reserve(m_predictedMeasurements);
    }
    inline void startStopWatch()
    {
        if (m_measurements.size() < m_maxMeasurements) {
            m_running = true;
            m_curTStart = high_resolution_clock::now();
        }
    }
    inline void stopStopWatch()
    {
        if (m_measurements.size() < m_maxMeasurements && m_running) {
            auto tEnd = high_resolution_clock::now();
            TimePair t;
            t.tStart = m_curTStart;
            t.tEnd = tEnd;
            m_measurements.push_back(t);
        }
    }
    inline int evaluateAverage()
    {
        int totalT = 0;
        for (TimePair& t : m_measurements) {
            totalT += duration_cast<microseconds>(t.tEnd - t.tStart).count();
        }
        if (m_measurements.size() > 0) {
            return totalT / m_measurements.size();
        }
        else {
            return 0;
        }
    }
    inline int lastMeasurement()
    {
        if (m_measurements.size() > 0) {
            TimePair& t = m_measurements.back();
            return duration_cast<microseconds>(t.tEnd - t.tStart).count();
        }
        return -1;
    }
};
} // namespace Util