
// #include "ra/Neural_net.hpp"
#include "ra/espcn.hpp"
#include <iostream>
int main (int argc, char** argv)
{

    ra::networks::espcn my_espcn;
    // std::vector <float> a;
    // for (int i=0; i < 900 ; ++i)
    //     a.push_back(i);
    std::string address;
    if (argc ==2)    
            
        address = argv[1];

    else
    {
        throw std::runtime_error("Please give the address to the weights");
    }

    

    my_espcn.read_input_image(std::cin);
    my_espcn.load_weights(address);
    my_espcn.upscal_image();







    return 0;
}