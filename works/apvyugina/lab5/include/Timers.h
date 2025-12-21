#include <chrono>
#include <stack>

using namespace std::chrono;

class Timer
{
public:
    std::stack<high_resolution_clock::time_point> tictoc_stack;

    void tic()
    {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        tictoc_stack.push(t1);
    }

    double toc(std::string msg = "", bool flag = true)
    {   
        duration<double, std::milli> diff = high_resolution_clock::now() - tictoc_stack.top();
        if(msg.size() > 0){
            if (flag)
                printf("%s time elapsed: %f ms\n", msg.c_str(), diff.count());
        }

        tictoc_stack.pop();
        return diff.count();
    }
    void reset()
    {
        tictoc_stack = std::stack<high_resolution_clock::time_point>();
    }
};