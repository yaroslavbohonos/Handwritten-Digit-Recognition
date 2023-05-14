#pragma once
#include <iostream>
#include <chrono>

struct Timer
{
    Timer (const char* label)
    {
        m_start = std::chrono::high_resolution_clock::now();
        m_label = label;
    }
 
    ~Timer ()
    {
        std::chrono::duration<float> seconds = std::chrono::high_resolution_clock::now() - m_start;
        printf("%s%0.2f seconds\n", m_label, seconds.count());
    }
 
    std::chrono::high_resolution_clock::time_point m_start;
    const char* m_label;
};