#include <iostream>
#include "mainloop.hpp"

using namespace std;

int main()
{
    cout << "Experiment start!" << endl;
    MainLoop mainLoop;
    mainLoop.Run();

    return 0;
}
