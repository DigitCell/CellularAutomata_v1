#include "cudasolver.hpp"
#include "tinycolormap.hpp"
#include <math.h>
//#include "transferViaPBO.hpp"
#include <helper_gl.h>

const unsigned kTextureSize = 512u;
const unsigned kTexturePixels = kTextureSize * kTextureSize;

__forceinline__ __host__ __device__ unsigned texturePixelIndex(unsigned x, unsigned y)
{
    return x + kTextureSize * y;
}

__device__ float clamp2(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ int clamp2(int x, int a, int b)
{
    return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b)
{
    r = clamp2(r, 0.0f, 255.0f);
    g = clamp2(g, 0.0f, 255.0f);
    b = clamp2(b, 0.0f, 255.0f);
    return (int(b) << 16) | (int(g) << 8) | int(r);
}


__device__
/// Returns the unit length vector for the given angle (in radians).
inline float2 VecFromAngle(const float a)
{
    return make_float2(cosf(a), sinf(a));
}

__device__
/// Returns the unit length vector for the given angle (in radians).
inline int2 intVecFromAngle(const float a, int dist)
{
    return make_int2((int)(cosf(a)*dist), (int)(sinf(a)*dist));
}


__device__ inline unsigned int mapPosCorrection(int x, int y, unsigned int width, unsigned int height)
{
    unsigned int xf=((x % width) + width)  % width;
    unsigned int yf=((y % height)+ height) % height;
    return yf*height+xf;
}

__device__ __host__ inline unsigned int mapPosCorrection2(int x, int y, unsigned int width, unsigned int height)
{
    unsigned int xf=(x+width)  % width;
    unsigned int yf=(y+height) % height;
    if(y<0 or y>=height){
        xf=width-xf;
    }
    if(x<0 or x>=width){
        yf=height-yf;
    }
    return yf*height+xf;
}


__device__ unsigned complementRGB_d(unsigned color)
{
    const unsigned char r = 255 - ((color >> 24) & 0xff);
    const unsigned char g = 255 - ((color >> 16) & 0xff);
    const unsigned char b = 255 - ((color >> 8) & 0xff);
    const unsigned char a = color & 0xff;
    return (r << 24) + (g << 16) + (b << 8) + a;
}


__global__ void Laplace(PointCA *T_old, PointCA *T_new, CudaNumberGenerator numberGen, const CudaParams * params, int tick)
{
    // compute the "i" and "j" location of the node point
    // handled by this thread

    const int NX = 512;      // mesh size (number of node points along X)
    const int NY = 512;

    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int j = blockIdx.y * blockDim.y + threadIdx.y ;

    // get the natural index values of node (i,j) and its neighboring nodes
                                //                                               N
    int P = mapPosCorrection(i  , j  , NX, NY);       // node (i,j)              |
    int N = mapPosCorrection(i  , j+1, NX, NY);       // node (i,j+1)            |
    int S = mapPosCorrection(i  , j-1, NX, NY);       // node (i,j-1)     W ---- P ---- E
    int E = mapPosCorrection(i+1, j  , NX, NY);       // node (i+1,j)            |
    int W = mapPosCorrection(i-1, j  , NX, NY);       // node (i-1,j)            |
                                //                                               S

    int NE = mapPosCorrection(i+1  , j+1  , NX, NY);       // node (i,j+1)            |
    int SW = mapPosCorrection(i-1  , j-1  , NX, NY);       // node (i,j-1)     W ---- P ---- E
    int EN = mapPosCorrection(i+1  , j-1  , NX, NY);       // node (i+1,j)            |
    int WS=  mapPosCorrection(i-1  , j+1  , NX, NY);       // node (i-1,j)            |
/*
    // only update "interior" node points
    if(i>0 && i<NX-1 && j>0 && j<NY-1) {
        T_new[P] = 0.25*( T_old[E] + T_old[W] + T_old[N] + T_old[S] );
    }
    */
    // T_new[P] = 0.25f*( T_old[E] + T_old[W] + T_old[N] + T_old[S] );
    float k=1.0f/8.0f;
    T_new[P].value = k*( T_old[E].value + T_old[W].value + T_old[N].value + T_old[S].value +
                   T_old[NE].value + T_old[SW].value + T_old[EN].value + T_old[WS].value );
    if( T_new[P].value>1.0f)
         T_new[P].value=1.0f;
}

__global__ void Laplace2(PointCA *T_old, PointCA *T_new, CudaNumberGenerator numberGen, const CudaParams * params, int tick)
{
    // compute the "i" and "j" location of the node point
    // handled by this thread

    const int NX = 512;      // mesh size (number of node points along X)
    const int NY = 512;

    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int j = blockIdx.y * blockDim.y + threadIdx.y ;

    float result=0;

    int Pd = mapPosCorrection(i  , j  , NX, NY);       // node (i,j)              |
    int Nd = mapPosCorrection(i  , j+1, NX, NY);       // node (i,j+1)            |
    int Sd = mapPosCorrection(i  , j-1, NX, NY);       // node (i,j-1)     W ---- P ---- E
    int Ed = mapPosCorrection(i+1, j  , NX, NY);       // node (i+1,j)            |
    int Wd = mapPosCorrection(i-1, j  , NX, NY);       // node (i-1,j)            |
                                //                                               S

    int NEd = mapPosCorrection(i+1  , j+1  , NX, NY);       // node (i,j+1)            |
    int SWd = mapPosCorrection(i-1  , j-1  , NX, NY);       // node (i,j-1)     W ---- P ---- E
    int ENd = mapPosCorrection(i+1  , j-1  , NX, NY);       // node (i+1,j)            |
    int WSd=  mapPosCorrection(i-1  , j+1  , NX, NY);       // node (i-1,j)            |

    result= ( T_old[Ed].value + T_old[Wd].value + T_old[Nd].value + T_old[Sd].value +
                   T_old[NEd].value + T_old[SWd].value + T_old[ENd].value + T_old[WSd].value );

    float k=result/8.0f;

    float deltaPlus=  numberGen.random(10)*M_PI/params->angle_div;
    int P =  mapPosCorrection(i  , j , NX, NY);

    T_new[P].status=T_old[P].status+0.3*numberGen.random(2);

    float angle=T_old[P].angle;
    if(T_old[P].status>params->status_border)
    {
        int genNumber=numberGen.random(100);
        if(genNumber>50)
            angle=T_old[P].angle+deltaPlus;
       // else if (genNumber<=50 )
       //     angle=T_old[P].angle-deltaPlus;
        T_new[P].status=numberGen.random(params->status_border-5);
    }

    int2 direction=intVecFromAngle(angle,numberGen.random(params->dist_1));
    int2 direction2=intVecFromAngle(angle,numberGen.random(params->dist_2));

    T_new[P].angle=angle;
    T_new[P].direction=direction;

    int P2=T_old[mapPosCorrection(i+direction.x  , j+direction.y, NX, NY)].value;
    int P3=0;
    //if(T_old[P].status<5)
     P3=sqrtf((T_old[mapPosCorrection(i+direction2.x  , j+direction2.y, NX, NY)].value));


    if(k>params->k1)
      T_new[P].value=k*params->k2;
    else if(k>params->k3)
       T_new[P].value=(params->k31*k+params->k32*P2+params->k33*P3);
    else
       T_new[P].value=k*0.15f;

    if(T_new[P].status>params->status_max)
         T_new[P].status=numberGen.random(params->status_border/2.0f);

    if( T_new[P].value>1.0f)
         T_new[P].value=1.0f;

    if( T_new[P].value<0.0011f)
         T_new[P].value=0.0011f;


     //T_new[P]=T_new[P]*0.85f;
}


__global__ void Laplace3(PointCA *T_old, PointCA *T_new, CudaNumberGenerator numberGen, const CudaParams * params, int tick)
{
    // compute the "i" and "j" location of the node point
    // handled by this thread

    const int NX = 512;      // mesh size (number of node points along X)
    const int NY = 512;

    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int j = blockIdx.y * blockDim.y + threadIdx.y ;

    float result=0;

    result+= 2*T_old[mapPosCorrection(i  , j+1  , NX, NY)].value;
    result+= T_old[mapPosCorrection(i  , j+2  , NX, NY)].value;

    result+= 2*T_old[mapPosCorrection(i  , j-1  , NX, NY)].value;
    result+= T_old[mapPosCorrection(i  , j-2  , NX, NY)].value;

    result+= 2*T_old[mapPosCorrection(i+1  , j  , NX, NY)].value;
    result+= T_old[mapPosCorrection(i+2  , j  , NX, NY)].value;

    result+= 2*T_old[mapPosCorrection(i-1  , j  , NX, NY)].value;
    result+= T_old[mapPosCorrection(i-2  , j  , NX, NY)].value;

    result+= T_old[mapPosCorrection(i-1  , j+1  , NX, NY)].value;
    result+= T_old[mapPosCorrection(i+1  , j+1  , NX, NY)].value;

    result+= T_old[mapPosCorrection(i-1  , j-1  , NX, NY)].value;
    result+= T_old[mapPosCorrection(i+1  , j-1  , NX, NY)].value;

     float k=result/16.0f;

/*
    // only update "interior" node points
    if(i>0 && i<NX-1 && j>0 && j<NY-1) {
        T_new[P] = 0.25*( T_old[E] + T_old[W] + T_old[N] + T_old[S] );
    }
    */
    // T_new[P] = 0.25f*( T_old[E] + T_old[W] + T_old[N] + T_old[S] );

    float deltaPlus=  M_PI/params->angle_div;
    int P =  mapPosCorrection(i  , j , NX, NY);

     T_new[P].status=T_old[P].status+1;

    //int P2 =  T_old[mapPosCorrection(i+(numberGen.random(3)-1)  , j+(numberGen.random(3)-1), NX, NY)].value;
    //int P3 =  T_old[mapPosCorrection(i+(numberGen.random(6)-3)  , j+(numberGen.random(6)-3), NX, NY)].value;
    float angle=T_old[P].angle;
    if(T_old[P].status>params->status_border)
    {
        int genNumber=numberGen.random(100);
        if(genNumber>50)
            angle=T_old[P].angle+deltaPlus;
        else if (genNumber<=50 )
            angle=T_old[P].angle-deltaPlus;

        T_new[P].status=numberGen.random(params->status_border-5);

    }


    int2 direction=intVecFromAngle(angle,numberGen.random(params->dist_1));
    int2 direction2=intVecFromAngle(angle,numberGen.random(params->dist_2));

    T_new[P].angle=angle;
    T_new[P].direction=direction;

    int P2=T_old[mapPosCorrection(i+direction.x  , j+direction.y, NX, NY)].value;
    int P3=0;
    //if(T_old[P].status<5)
     P3=T_old[mapPosCorrection(i+direction2.x  , j+direction2.y, NX, NY)].value;


    if(k>params->k1)
      T_new[P].value=k*params->k2;
    else if(k>params->k3)
       T_new[P].value=(params->k31*k+params->k32*P2+params->k33*P3);
    else
       T_new[P].value=(k*0.15f);



    if(T_new[P].status>params->status_max)
         T_new[P].status=numberGen.random(params->status_border);

    if( T_new[P].value>1.5f)
         T_new[P].value=1.5f;

    if( T_new[P].value<0.011f)
         T_new[P].value=0.011f;


     //T_new[P]=T_new[P]*0.85f;
}

__global__ void Laplace_sync(PointCA *T_old, PointCA *T_new)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x ;
    uint j = blockIdx.y * blockDim.y + threadIdx.y ;

    uint z=i*kTextureSize+j;

    if(z<kTexturePixels)
   {
        T_old[z]= T_new[z];
    }

}

CudaSolver::CudaSolver()
{

     setTexture(0u, makeJorgeImage(kTextureSize));
     m_cuda_rast_params.resize(1u);
     glGenBuffers(1, &m_pbo);

     setScreenSize(512u, 512u);

     h_colorsMap=(unsigned*)malloc(sizeof(int)*colorMapNumbers);
     for(int i=0; i<colorMapNumbers;i++)
     {
         float value=(float)i/colorMapNumbers;
         const tinycolormap::Color color = tinycolormap::GetColor(value, tinycolormap::ColormapType::Viridis);
        // h_colorsMap[i]=complementRGB(unsigned(colorMapNumbers*color.b()) << 16) | (unsigned(colorMapNumbers*color.g()) << 8) | unsigned(colorMapNumbers*color.r());
         h_colorsMap[i]=cRGB(unsigned(colorMapNumbers*color.r()),
                             unsigned(colorMapNumbers*color.g()),
                             unsigned(colorMapNumbers*color.b()),
                             254);
     }
      cudaMalloc(&d_colorsMap,sizeof(unsigned)*colorMapNumbers);
      checkCudaCall(cudaMemcpy(d_colorsMap, h_colorsMap, sizeof(unsigned)*colorMapNumbers, cudaMemcpyHostToDevice));

    // allocate storage space on the GPU

    m_texWidth=kTextureSize;
    m_texHeight=kTextureSize;

    PointCA *T = new PointCA [m_texHeight*m_texWidth];

    // initialize array on the host
    InitializeT(T);

    cudaMalloc(&_T1,m_texHeight*m_texWidth*sizeof(PointCA));
    cudaMalloc(&_T2,m_texHeight*m_texWidth*sizeof(PointCA));

    // copy (initialized) host arrays to the GPU memory from CPU memory
    checkCudaCall(cudaMemcpy(_T1,T,m_texHeight*m_texWidth*sizeof(PointCA),cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(_T2,T,m_texHeight*m_texWidth*sizeof(PointCA),cudaMemcpyHostToDevice));

      delete T;

    cudaRND.init(100000000);

}


void CudaSolver::CudaReInitSolver()
{
    m_texWidth=kTextureSize;
    m_texHeight=kTextureSize;

    PointCA *T = new PointCA [m_texHeight*m_texWidth];

    // initialize array on the host
    InitializeT(T);




    // copy (initialized) host arrays to the GPU memory from CPU memory
    checkCudaCall(cudaMemcpy(_T1,T,m_texHeight*m_texWidth*sizeof(PointCA),cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(_T2,T,m_texHeight*m_texWidth*sizeof(PointCA),cudaMemcpyHostToDevice));

      delete T;
}

void CudaSolver::CudaSolverStep(CudaParams& params)
{

    params.height = m_texHeight;
    params.width =  m_texWidth;
    params.pixelsNumber= m_texPixels;
    params.texture = m_cuda_textures.ptr();
    params.colorsMap=d_colorsMap;


    checkCudaCall(cudaMemcpy(m_cuda_rast_params.ptr(), &params, sizeof(CudaParams), cudaMemcpyHostToDevice));
    const int tc = m_threadsperblock;
    const int bc = (m_texWidth + tc - 1) / tc;
    m_timer.start();
    //cuda_solver << <bc, tc >> > (m_cuda_rast_params.ptr());

    dim3 threadsperBlock(32, 32);
    dim3 numBlocks1((m_texWidth + threadsperBlock.x - 1) / threadsperBlock.x,
                          (m_texHeight + threadsperBlock.y - 1) / threadsperBlock.y);

    Laplace2<<<numBlocks1, threadsperBlock >>>(_T1,_T2, cudaRND, m_cuda_rast_params.ptr(), tick);   // update T1 using data stored in T2
    cudaDeviceSynchronize();
    Laplace_sync<<<numBlocks1, threadsperBlock >>>(_T1,_T2);   // update T2 using data stored in T1
    //checkCudaCall(cudaMemcpy(_T1,_T2,m_texHeight*m_texWidth*sizeof(float),cudaMemcpyDeviceToDevice));
    cuda_draw<< <numBlocks1, threadsperBlock >> > (m_cuda_rast_params.ptr(),_T1, cudaRND);
    cudaDeviceSynchronize();
    m_timer.stop();
    checkCudaCall(cudaGetLastError());

    tick++;
}


__global__ void cuda_clean(const CudaParams * params)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < params->width && y < params->height)
    {
        // clear particle world
        const unsigned int pixelIndex=y*kTextureSize+x;
        params->texture[pixelIndex]= params->colorsMap[1];
    }
}

__global__ void cuda_draw(const CudaParams * params, PointCA *T,  CudaNumberGenerator numberGen )
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < params->width && y < params->height)
    {
        const unsigned int pixelIndex=y*kTextureSize+x;
        float pcolor=T[pixelIndex].value;
        params->texture[pixelIndex]= params->colorsMap[(int)(pcolor*175.0f)];
/*
        if(numberGen.random(10000)>9991)
        {
            T[pixelIndex]=0.05;//+numberGen.random(0.5f);
        }
        */

    }
}



void CudaSolver::InitializeT(PointCA *TEMPERATURE)
{
    for(int i=0;i<m_texWidth;i++) {
        for(int j=0;j<m_texHeight;j++) {
            int index = i + j*m_texWidth;
            TEMPERATURE[index].value=0.05;
            TEMPERATURE[index].direction=make_int2(0,0);

            if(i>225 and i<230 and j>225 and j<230)
            {
                TEMPERATURE[index].value=1.0;
                TEMPERATURE[index].direction=make_int2(1,0);
            }
/*
            if(i>225+100 and i<230+100 and j>225+100 and j<230+100)
                TEMPERATURE[index]=1.0;

            if(i>225-100 and i<230-100 and j>225-100 and j<230-100)
                TEMPERATURE[index]=1.0;

            if(i>100+200 and i<150+200 and j>100+200 and j<150+200)
                TEMPERATURE[index]=1.0;
                */
        }
    }


}


void CudaSolver::transferViaPBO(unsigned * cudascreen, sf::Texture& texture, unsigned pbo)
{
    if(glXGetCurrentContext() == NULL)
        return;

    const uint x = kTextureSize;
    const uint y = kTextureSize;
    const uint size = kTexturePixels * 4;

    void * ptr = 0x0;
    size_t buffsize = 0u;
    checkCudaCall(cudaGraphicsMapResources(1, &resCuda));
    checkCudaCall(cudaGraphicsResourceGetMappedPointer(&ptr, &buffsize, resCuda));
    checkCudaCall(cudaMemcpy(ptr, cudascreen, buffsize, cudaMemcpyDeviceToDevice));
    checkCudaCall(cudaGraphicsUnmapResources(1, &resCuda));

    sf::Texture::bind(&texture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        x, y, 0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        0x0
    );
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


}


void CudaSolver::transferViaPBOCreateBuffer(unsigned * cudascreen, sf::Texture& texture, unsigned pbo)
{
    if(glXGetCurrentContext() == NULL)
        return;

    const int x = static_cast<int>(texture.getSize().x);
    const int y = static_cast<int>(texture.getSize().y);
    const unsigned size = texture.getSize().x * texture.getSize().y * 4;

    cudaGraphicsResource * res;
    checkCudaCall(cudaGraphicsGLRegisterBuffer(&res, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
    void * ptr = 0x0;
    size_t buffsize = 0u;
    checkCudaCall(cudaGraphicsMapResources(1, &res));
    checkCudaCall(cudaGraphicsResourceGetMappedPointer(&ptr, &buffsize, res));
    checkCudaCall(cudaMemcpy(ptr, cudascreen, buffsize, cudaMemcpyDeviceToDevice));
    checkCudaCall(cudaGraphicsUnmapResources(1, &res));
    checkCudaCall(cudaGraphicsUnregisterResource(res));

    sf::Texture::bind(&texture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        x, y, 0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        0x0
    );
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}
void CudaSolver::initPBO(unsigned * cudascreen, sf::Texture& texture, unsigned pbo)
{
    if(glXGetCurrentContext() == NULL)
        return;

    const int x = static_cast<int>(texture.getSize().x);
    const int y = static_cast<int>(texture.getSize().y);
    const unsigned size = texture.getSize().x * texture.getSize().y * 4;


    checkCudaCall(cudaGraphicsGLRegisterBuffer(&resCuda, pbo, cudaGraphicsRegisterFlagsWriteDiscard));

}



CudaSolver::~CudaSolver()
{
    glDeleteBuffers(1, &m_pbo);
}

bool CudaSolver::getUsePbo() const
{
      return m_usepbo;
}

void CudaSolver::setUsePbo(bool usepbo)
{
    m_usepbo = usepbo;
}


__device__ unsigned cuda_screenPixelIndex(const CudaParams * params, unsigned x, unsigned y)
{
    return x + params->width * y;
}

__device__ const unsigned * cuda_getTexture(const CudaParams * params, unsigned num)
{
    if(num < params->pixelsNumber)
        return params->texture + num * kTexturePixels;

    return params->texture; //jorge
}



#define byteswap(v)(((v>>24)&0xff)|((v<<8)&0xff0000)|((v>>8)&0xff00)|((v<<24)&0xff000000))
void CudaSolver::setTexture(unsigned texnum, const sf::Image &img)
{
    if(img.getSize() != sf::Vector2u(kTextureSize, kTextureSize))
        return;

    unsigned tex[kTexturePixels];
    for(int x = 0; x < kTextureSize; ++x)
        for(int y = 0; y < kTextureSize; ++y)
            tex[texturePixelIndex(x, y)] = byteswap(img.getPixel(x, y).toInteger());

    if((texnum * kTexturePixels) >= m_cuda_textures.size())
        m_cuda_textures.resize((texnum + 1u) * kTexturePixels);

    checkCudaCall(cudaMemcpy(m_cuda_textures.ptr() + texnum * kTexturePixels, tex, sizeof(unsigned) * kTexturePixels, cudaMemcpyHostToDevice));
}
#undef byteswap


void CudaSolver::downloadImage(sf::Texture &texture)
{
    if(texture.getSize() != sf::Vector2u(m_texWidth, m_texHeight))
            texture.create(m_texWidth, m_texHeight);

    if(m_usepbo)
    {
        transferViaPBO(m_cuda_textures.ptr(), texture, m_pbo);
    }
    else
    {
        checkCudaCall(cudaMemcpy(m_texture.data(), m_cuda_textures.ptr(), m_texPixels * 4u, cudaMemcpyDeviceToHost));
        texture.update(reinterpret_cast<sf::Uint8*>(m_texture.data()));
    }
}

void CudaSolver::InitCudaImage(sf::Texture &texture)
{
    if(texture.getSize() != sf::Vector2u(m_texWidth, m_texHeight))
            texture.create(m_texWidth, m_texHeight);

    if(m_usepbo)
    {
        initPBO(m_cuda_textures.ptr(), texture, m_pbo);
    }

}


void CudaSolver::setScreenSize(unsigned width, unsigned height)
{
    if(m_texWidth == width && m_texHeight == height)
          return;

    height = height - (height % 2); //only even height allowed
    m_texWidth= width;
    m_texHeight = height;
    m_texPixels = width * height;
    m_texture.assign(m_texPixels, 0x7f7f7fff);

    m_cuda_texture.resize(m_texPixels);
    checkCudaCall(cudaMemset(m_cuda_texture.ptr(), 0xff, m_cuda_texture.bytesize()));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * m_texPixels, 0x0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0u);
}






