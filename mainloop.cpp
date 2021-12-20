#include "mainloop.hpp"


void MainLoop::Run()
{
    srand (time(NULL));
    Init();

    int iter_number=0;
    int pause=0;
    int usespace=0;

    while (graph2d.window.isOpen()) {

       graph2d.updateFrame(params, 0);

       if(graph2d.changeButtonPress)
       {
           cudaSolver.downloadImage(graph2d.texCuda);
           //graph2d.spriteCuda.setTexture(graph2d.texCuda);

          // graph2d.spriteCuda.setPosition(sf::Vector2f(500,50));
          // graph2d.spriteCuda.setScale(1.0f,1.0f);
       }

       if(graph2d.m_framecounter>5)
       {
           cudaSolver.CudaSolverStep(params);
           cudaSolver.downloadImage(graph2d.texCuda);
       }

       if(graph2d.changeButtonPress)
       {
           cudaSolver.CudaReInitSolver();
       }
    }
    graph2d.Close(0);
}

void MainLoop::Init()
{
    graph2d.initPictures();
    cudaSolver.InitCudaImage(graph2d.texCuda);

    params.k1=0.395;


}
