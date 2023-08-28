#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <cassert>

#define PROFILE_VN_CONCAT_INNR(a, b) a##b
#define PROFILE_VN_CONCAT(a, b) PROFILE_VN_CONCAT_INNR(a, b)
#define PROFILE(x) auto PROFILE_VN_CONCAT(_profiler, __COUNTER__) = g_FrameProfiler.push_caller(x)
#define PROFILE_PREC(x) auto PROFILE_VN_CONCAT(_profiler, __COUNTER__) = g_PreComputeProfiler.push_caller(x)
#define PROFILE_STEP(x) auto PROFILE_VN_CONCAT(_profiler, __COUNTER__) = g_StepProfiler.push_caller(x)
#define PROFILE_X(p, x) auto PROFILE_VN_CONCAT(_profiler, __COUNTER__) = p.push_caller(x)

namespace Util {

struct DestructorFuncCaller {
    std::function<void()> func;

    ~DestructorFuncCaller() { func(); }
};

struct Profiler {
    struct Section {
        std::string name;
        std::vector<Section> sections;
        Section* parent = nullptr;

        size_t _numExec = 0;
        double _sumTime = 0.0;
        double _beginTime = 0.0;
        double _avgTime = 0.0;
        double _lastTime = 0.0;

        Section& find(std::string_view _n)
        {
            assert(_n.length() > 0);
            for (Section& s : sections) {
                if (s.name == _n)
                    return s;
            }
            Section& sec = sections.emplace_back();
            sec.name = _n;
            return sec;
        }
        void reset()
        {
            _sumTime = 0;
            _numExec = 0;
            _beginTime = 0;
            _avgTime = 0;
            _lastTime = 0;
            for (Section& s : sections) {
                s.reset();
            }
        }
    };
    std::thread::id m_LocalThreadId;

    Section m_RootSection;
    Section* m_CurrentSection = &m_RootSection;
    Section* sectionToBeClear = nullptr; // clear section should after it's pop().

public:
    static double nanoseconds()
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

    void push(std::string_view name)
    {
        if (m_RootSection.sections.size() == 0) {
            m_LocalThreadId = std::this_thread::get_id();
        }
        else {
            assert(m_RootSection.sections.size() == 1);
            assert(m_LocalThreadId == std::this_thread::get_id());
        }

        Section& sec = m_CurrentSection->find(name);

        sec.parent = m_CurrentSection; // if (sec.parent == nullptr)
        sec._beginTime = nanoseconds();

        m_CurrentSection = &sec;
    }

    void pop()
    {
        if (sectionToBeClear == m_CurrentSection) {
            sectionToBeClear = nullptr;
            auto parent = m_CurrentSection->parent;
            m_CurrentSection->reset();
            m_CurrentSection = parent;
            return;
        }

        Section& sec = *m_CurrentSection;

        double dur = (nanoseconds() - sec._beginTime) / 1e9;
        sec._numExec++;
        sec._sumTime += dur;
        sec._lastTime = dur;
        sec._avgTime = sec._sumTime / sec._numExec;

        m_CurrentSection = m_CurrentSection->parent;
    }

    Section& GetRootSection()
    {
        if (m_RootSection.sections.empty()) return m_RootSection; // temporary solution when no section recorded.
        assert(m_RootSection.sections.size() == 1);
        return m_RootSection.sections.at(0);
    }

    // when we want reset/clear profiler data, we cannot just direct clear,
    // it consist push/pop,. clear should be after last pop.
    void laterClearRootSection()
    {
        if (m_CurrentSection == &m_RootSection) { // just clear directly.
            m_RootSection.reset();
        }
        else {
            // Delay clear after last pop.
            sectionToBeClear = &GetRootSection();
        }
    }

    // clang-format off
    [[nodiscard]] 
    DestructorFuncCaller push_caller(std::string_view name)
    {
        push(name);
        return DestructorFuncCaller{ [this]() {pop();} };
    }
    // clang-format on
};

}

extern Util::Profiler g_FrameProfiler;
extern Util::Profiler g_StepProfiler;
extern Util::Profiler g_PreComputeProfiler;