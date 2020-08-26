#pragma once

#include<chrono>

class CTimer
{
public:
	CTimer() {};

	inline void start()
	{
		m_tmBegin = std::chrono::high_resolution_clock::now();
	}

	inline long long stop()
	{
		m_tmEnd = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::milliseconds>(m_tmEnd - m_tmBegin).count();
	}

	inline long long stop_microseconds()
	{
		m_tmEnd = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::microseconds>(m_tmEnd - m_tmBegin).count();
	}

	inline long long stop_seconds()
	{
		m_tmEnd = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::seconds>(m_tmEnd - m_tmBegin).count();
	}

	inline long long stop_nanoseconds()
	{
		m_tmEnd = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::nanoseconds>(m_tmEnd - m_tmBegin).count();
	}

	inline long long timeSecs() const
	{
		return std::chrono::duration_cast<std::chrono::seconds>(m_tmEnd - m_tmBegin).count();
	}

	inline long long timeMilliSecs() const
	{
		return std::chrono::duration_cast<std::chrono::milliseconds>(m_tmEnd - m_tmBegin).count();
	}

	inline long long timeMicroSecs() const
	{
		return std::chrono::duration_cast<std::chrono::microseconds>(m_tmEnd - m_tmBegin).count();
	}

	inline long long timeNanoSecs() const
	{
		return std::chrono::duration_cast<std::chrono::nanoseconds>(m_tmEnd - m_tmBegin).count();
	}



private:
	std::chrono::high_resolution_clock::time_point m_tmBegin, m_tmEnd;
};