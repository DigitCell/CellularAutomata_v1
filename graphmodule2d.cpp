#include "graphmodule2d.hpp"


void Graphmodule2D::Gui(CudaParams& params, int tick)
{
    ImGui::Begin("Command window");
    ImGui::Text("Frame %019d", m_framecounter);


    ImGui::Text("GL_VENDOR   = %s", m_glvendor.c_str());
    ImGui::Text("GL_RENDERER = %s", m_glrenderer.c_str());

    ImGui::Text("FPS: %f\n", m_fpscounter.frame());

    changeButtonPress=ImGui::Button("Press to generate random");
    //generateButtonPress=ImGui::Button("Press to generate all");

    ImGui::SliderFloat("coeff k1 ", &params.k1, 0.1f , 1.0f);
    ImGui::SliderFloat("coeff k2 ", &params.k2, 0.5f , 1.0f);
    ImGui::SliderFloat("coeff k3 ", &params.k3, 0.01f , 0.1f);
    ImGui::SliderFloat("coeff k31 ", &params.k31, 0.7f , 1.0f);
    ImGui::SliderFloat("coeff k32 ", &params.k32, 0.5f , 1.3f);
    ImGui::SliderFloat("coeff k33 ", &params.k33, 0.1f , 1.3f);

    ImGui::SliderInt("status max ", &params.status_max, 1 , 125);
    ImGui::SliderInt("status border ", &params.status_border, 1, 125);
    ImGui::SliderFloat("angle div ", &params.angle_div, 2.0f , 36.0f);

    ImGui::SliderInt("dist 1 ", &params.dist_1, 0, 9);
    ImGui::SliderInt("dist 2 ", &params.dist_2, 0, 9);

    ImGui::End();

    ImGui::Begin("Pic window");
    ImGui::Image(texCuda, sf::Vector2f(300,300));
    ImGui::End();

}



void Graphmodule2D::updateFrame(CudaParams& params,int time)
{
    //Events
    sf::Event event;
    while (window.pollEvent(event)) {
       ImGui::SFML::ProcessEvent(event);

       if (event.type == sf::Event::Closed) {
           window.close();
       }

       switch(event.type)
       {
       case sf::Event::Closed:
           window.close();
           break;
       case sf::Event::Resized:
           window.setView(sf::View(sf::FloatRect(0.f, 0.f, event.size.width, event.size.height)));
           break;
       case sf::Event::KeyPressed:
           if(event.key.code == sf::Keyboard::Semicolon)
           {
               sf::Texture tex;
               tex.create(window.getSize().x, window.getSize().y);
               tex.update(window);

               char buff[128];
               std::snprintf(buff, 120, "sshot%09d.jpeg", m_framecounter);
               const auto img = tex.copyToImage();
               img.saveToFile("../screenshorts/"+std::string(buff));
               std::printf("Screenshot taken: %s\n", buff);
           }
           break;
       }//switch
    }

    //ImGui
    ImGui::SFML::Update(window, deltaClock.restart());
    Gui(params, 0);

    //Draw
    DrawFrame(0);

    //Display
    ImGui::SFML::Render(window);
    window.display();
}

void Graphmodule2D::clearFrame()
{
    window.clear();
}

void Graphmodule2D::DrawFrame(int time)
{
    clearFrame();
    //window.draw(sprite);
    window.draw(spriteCuda);
    ++m_framecounter;
}



void Graphmodule2D::Close(int time)
{
     ImGui::SFML::Shutdown();
}

//Pictures Load

void Graphmodule2D::initPictures()
{

    const sf::Color emptyColor(0,0,0,0);
    sf::Image tempImage;

    backGroundImage.create(imageHeight,imageWidth, emptyColor);
    backGroundImage.loadFromFile("../textures/side3_blue512.png");

    tex.loadFromImage(backGroundImage);
    sprite.setTexture(tex);

    sprite.setPosition(sf::Vector2f(50,50));
    sprite.setScale(0.5f,0.5f);

    texCuda.loadFromImage(backGroundImage);
    spriteCuda.setTexture(texCuda);

    spriteCuda.setPosition(sf::Vector2f(500,50));
    spriteCuda.setScale(1.0f,1.0f);


}
