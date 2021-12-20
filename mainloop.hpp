#ifndef MAINLOOP_HPP
#define MAINLOOP_HPP

#pragma once
#include "graphmodule2d.hpp"
#include "cudasolver.hpp"

#include "HelperStructs.hpp"

class MainLoop
{
public:
    MainLoop():
       graph2d(1100,1000)
    {

    };

    void Run();

    void Init();
    void Shutdown();
    void Update();

    Graphmodule2D graph2d;
    CudaSolver cudaSolver;

    CudaParams params;


};

#endif // MAINLOOP_HPP
