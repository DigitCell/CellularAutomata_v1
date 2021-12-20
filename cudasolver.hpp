#ifndef CUDASOLVER_HPP
#define CUDASOLVER_HPP

#pragma once

#include <SFML/Graphics/Image.hpp>
#include <SFML/Window/Keyboard.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Graphics/Texture.hpp>

#include "CudaAutoBuffer.hpp"
#include "CudaEventTimer.hpp"


#include <SFML/Graphics/Texture.hpp>
#include <cstring>
#include "checkCudaCall.hpp"
#include "cuda_gl_interop.h"


#include <cmath>

#include "cuda_runtime_api.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "checkCudaCall.hpp"
#include "jorge.hpp"

#include "cudahelper.hpp"

#include "HelperStructs.hpp"

struct PointCA
{
    float value=0.0f;
    int2 direction=make_int2(0,0);
    int  color=0;
    float  status=0;
    float old_value=0.0f;
    float angle=0;
};

class CudaSolver
{
public:
    CudaSolver();
    ~CudaSolver();


    bool getUsePbo() const;
    void setUsePbo(bool usepbo);
;

    void setScreenSize(unsigned width, unsigned height);
    void setTexture(unsigned texnum, const sf::Image& img);
    void downloadImage(sf::Texture& texture);


    std::vector<unsigned> m_texture;
    unsigned m_texWidth;
    unsigned m_texHeight;
    unsigned m_texPixels;

    cudaGraphicsResource * resCuda;



    CudaAutoBuffer<unsigned> m_cuda_textures;
    CudaAutoBuffer<unsigned> m_cuda_texture;
    CudaAutoBuffer<CudaParams> m_cuda_rast_params;

    CudaNumberGenerator cudaRND;

    int m_threadsperblock = 64;
    std::string m_name;
    CudaEventTimer m_timer;
    unsigned m_pbo = 0u;
    bool m_usepbo = true;

    int colorMapNumbers=254;
    unsigned* h_colorsMap;
    unsigned* d_colorsMap;


    PointCA *T;          // pointer to host (CPU) memory
    PointCA *_T1, *_T2;  // pointers to device (GPU) memory


    void transferViaPBO(unsigned * cudascreen, sf::Texture& texture, unsigned pbo);
    void initPBO(unsigned *cudascreen, sf::Texture &texture, unsigned pbo);
    void InitCudaImage(sf::Texture &texture);

    void transferViaPBOCreateBuffer(unsigned *cudascreen, sf::Texture &texture, unsigned pbo);

    void CudaSolverStep(CudaParams& params);
    void InitializeT(PointCA *TEMPERATURE);

    int tick=0;

    void CudaReInitSolver();
};

__global__ void cuda_clean(const CudaParams * params);

__global__ void cuda_draw(const CudaParams * params, PointCA *T, CudaNumberGenerator numberGen);

__global__ void Laplace_sync(PointCA *T_old, PointCA *T_new);

#endif // CUDASOLVER_HPP
