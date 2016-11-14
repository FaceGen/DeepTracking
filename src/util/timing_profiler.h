#ifndef TIMING_PROFILER_
#define TIMING_PROFILER_
#include <vector>
#include <map>
#include <string>
#include <time.h>

//-----------------------------------------------------------------------------
class timing_profiler {

  public:
    timing_profiler() {time_pieces.clear(); history.clear(); };
    ~timing_profiler() {time_pieces.clear(); history.clear(); };
    
    // calibrate floor coordinate system from at least 3 points on a same plane which is parrallel to floor plane
    void reset(void);
    void update(std::string& name);
    float getTimePieceInMillisecend(std::string& name);
    char* getTimeProfileString(void);
    char* getSmoothedTimeProfileString(void);

private:
    unsigned long long starting_time_in_microsend;
    unsigned long long cur_time_in_microsend;

    char profile_string[10000];
    std::map<std::string, float> time_pieces;
    std::map<std::string, float> history;

};

unsigned long long GetCurrentMicroSecond();

#endif