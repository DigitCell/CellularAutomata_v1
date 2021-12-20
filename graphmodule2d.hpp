#ifndef GRAPHMODULE2D_HPP
#define GRAPHMODULE2D_HPP

#pragma once

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Graphics/CircleShape.hpp>

#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/RenderTexture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Image.hpp>

#include "FpsCounter.hpp"

#include <imgui.h>
#include <imgui-SFML.h>

#include <stdlib.h>

//#include <helper_gl.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include "vector"

#include "HelperStructs.hpp"

class Graphmodule2D
{

public:
    Graphmodule2D(int _scrwidth, int _scrheghts) :
        scr_width(_scrwidth),
        scr_height(_scrheghts),
        window(sf::VideoMode(scr_width, scr_height), "SFML CUDA experiments")
    {
        window.setFramerateLimit(60);
        ImGui::SFML::Init(window);

        m_glvendor = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
        m_glrenderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    };


   int scr_width=300;
   int scr_height=300;

   sf::RenderWindow window;

   sf::Clock deltaClock;
   sf::Clock m_guiclock;
   sf::Texture m_texture;
   FpsCounter m_fpscounter;

   sf::Clock m_movementclock;
   std::string m_glvendor;
   std::string m_glrenderer;
   int m_framecounter = 0;

   sf::Sprite sprite;
   sf::Texture tex;

   sf::Sprite spriteCuda;
   sf::Texture texCuda;


   void initPictures();
   void updateFrame(CudaParams& params,int time);
   void clearFrame();
   void DrawFrame(int time);
   void Gui(CudaParams& params,int tick);

   void Close(int time);


   unsigned int imageWidth=2300;
   unsigned int imageHeight=2300;

   sf::Image backGroundImage;

   bool changeButtonPress;
   bool generateButtonPress;

   int RandInt(int nMin, int nMax)
   {
       //return nMin + (int)((double)rand() / (RAND_MAX+1) * (nMax-nMin+1));
       return rand() % (nMax - nMin) + nMin + 1;
   }

};

#endif // GRAPHMODULE2D_HPP
