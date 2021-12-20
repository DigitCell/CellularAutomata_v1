#ifndef HELPERSTRUCTS_HPP
#define HELPERSTRUCTS_HPP

#pragma once

class CudaParams
{
public:

    int width;
    int height;
    int pixelsNumber;
    unsigned * texture;
    unsigned * colorsMap;


    float k1=0.395f;
    float k2=0.967f;

    float k3=0.0391f;

    float k31=0.9995f;
    float k32=0.95f;
    float k33=0.9f;

    int status_max=15;
    int status_border=7;

    float angle_div=7.0f;
    int dist_1=3;
    int dist_2=2;

};




#endif // HELPERSTRUCTS_HPP
