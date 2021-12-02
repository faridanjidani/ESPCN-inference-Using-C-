#include "../include/ra/espcn.hpp"


using namespace ra::networks;

espcn::espcn()
{
    auto input = nn1.Input(1); // input channel size
    auto x = nn1.Conv2D(input, 64, 5, 1, "same", "tanh");
    x = nn1.Conv2D(x, 32, 3, 1, "same", "tanh");
    x = nn1.Conv2D(x, 9, 3, 1, "same");    
    x = nn1.depth_to_space(x, 3);
}


void espcn::load_weights(std::string file)
{
    nn1.load_weights(file);
}


void espcn::read_input_image (std::istream& in)
{

    in >> std::noskipws;
    std::vector<char>input_temp_array;
    std::copy(std::istream_iterator<char>(in), std::istream_iterator<char>(), std::back_inserter(input_temp_array));
    cv::Mat in_image;
    cv::Mat ycbr;
    try
    {
        in_image  = cv::imdecode(cv::Mat(input_temp_array),1);
        input_height = in_image.rows;
        input_width =  in_image.cols;



        cv::cvtColor(in_image, ycbr, cv::COLOR_RGB2YCrCb);
   
    }
    catch(const std::exception& e)
    {
        throw std::runtime_error("input image not valid");
    }
    



    cv::split (ycbr, ycbr_split);
    if (ycbr_split[0].isContinuous()) {
    input_image.assign(ycbr_split[0].data, ycbr_split[0].data + ycbr_split[0].total()*ycbr_split[0].channels());
    } 
    else {
        for (int i = 0; i < ycbr_split[0].rows; ++i) {
            input_image.insert(input_image.end(), ycbr_split[0].ptr<float>(i), ycbr_split[0].ptr<float>(i)+ycbr_split[0].cols*ycbr_split[0].channels());
        }
    }

    std::transform(input_image.begin(), input_image.end(), input_image.begin(), [](float &c){ return c/255; });
    
}

void espcn::upscal_image()

{
    cv::Mat resized_ycbr_1;
    cv::Mat resized_ycbr_2;

    auto upscaled_image = nn1.predict(input_image, input_width, input_height);
    cv::resize(ycbr_split[1], resized_ycbr_1, cv::Size(), 3, 3, cv::INTER_CUBIC); // upscale 3x    
    cv::resize(ycbr_split[2], resized_ycbr_2, cv::Size(), 3, 3, cv::INTER_CUBIC); // upscale 3x   


    for (size_type i =0; i<upscaled_image.size(); i++)  //clipping
    {
        upscaled_image[i] *= 255.0;
        if (upscaled_image[i]>255)
            upscaled_image[i] = 255;
        else if (upscaled_image[i]<0)
             upscaled_image[i] = 0;

    }  

    const int size[2] = {input_height*3, input_width*3};
    cv::Mat Yc(2, size, CV_32FC1, upscaled_image.data());
    Yc.convertTo(Yc, CV_8UC1);
    std::vector <cv::Mat> channels = {Yc,resized_ycbr_1,resized_ycbr_2};
    cv::Mat merged;
    cv::merge(channels,merged); 
    cv::Mat rgb_image;
    cv::cvtColor(merged, rgb_image, cv::COLOR_YCrCb2RGB);

    // cv::imwrite("starry_night.jpg", rgb_image);
    std::vector<uchar> out_buffer;
    cv::imencode(".jpg",rgb_image,out_buffer  );
    
    for (size_type i=0 ; i<out_buffer.size(); i++)
        std::cout<<out_buffer[i] ;

}

