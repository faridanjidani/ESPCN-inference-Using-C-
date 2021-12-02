
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <cassert>
#include<stdexcept>
#include <cmath> 
namespace ra::neural_net {

    template <class T>
    class Neural_net
    {

    public:

        using value_type = T;

        // An unsigned integral type used to represent sizes.
        using size_type = std::size_t;
        
        enum class  layer_type // Different type of layers
        {
            Conv2D,
            depth_to_space,
            activation,
            Conv1D,
            SeparableConv2D
        };

        struct layer_conf 
        {
            layer_type type_of_layer;
            std::vector<std::vector<T> > weights;
            std::vector<T> bias;
            int filter_size;
            int stride;
            std::string padding;
            int output_channel;
            int input_channel;
            std::string input_layer;
            std::string activation;
        };

        Neural_net()
        {}
        Neural_net(const Neural_net&) = delete;
        Neural_net& operator=(const Neural_net&) = delete;
        Neural_net(Neural_net&&) = delete;
        Neural_net& operator=(Neural_net&&) = delete;
        ~Neural_net()
        {}

        std::string Input(int channel_size = 1)
        {
            layer_conf l_confg; 
            l_confg.output_channel = channel_size;
            l_confg.input_layer = "Dummy";
            std::string layer_name = "Input";
            layer_map[layer_name] =l_confg ;
            return layer_name;

        }

        std::string Conv2D(std::string input_name = "input", int output_channel =32, int filter_size =3, int stride =1, std::string padding= "same", std::string activation ="None")
        {
            layer_conf l_confg; 
            l_confg.type_of_layer = layer_type::Conv2D;
            l_confg.filter_size = filter_size;
            l_confg.stride = stride;
            l_confg.padding = padding;
            l_confg.output_channel = output_channel;
            l_confg.input_channel = (layer_map[input_name]).output_channel;
            l_confg.input_layer = input_name;
            l_confg.activation = activation;
            std::string layer_name = "conv_" + std::to_string(layer_number);
            layer_number = layer_number+1;
            layer_map[layer_name] =l_confg ;
            return layer_name;

        }

        std::string depth_to_space(std::string input_name = "input", int factor=3)
        {
            layer_conf l_confg; 
            l_confg.type_of_layer = layer_type::depth_to_space;
            l_confg.output_channel = factor * factor;
            l_confg.input_layer = input_name;
            std::string layer_name = "depth_to_space" + std::to_string(layer_number);
            layer_number = layer_number+1;
            layer_map[layer_name] =l_confg ;
            return layer_name;

        }
        void load_weights (std::string file)
        {
            std::ifstream infile(file, std::ios::in);

            

            for ( auto it= ++(layer_map.begin()); it!=layer_map.end(); ++it)
            {
                it->second.weights.resize(it->second.output_channel);       
                it->second.bias.resize(it->second.output_channel);                      
                for (size_type j=0; j<it->second.weights.size(); ++j)
                {
                    size_type vector_size = it->second.filter_size*it->second.filter_size*it->second.input_channel;
                    std::vector <T> temp_weight(vector_size);
                    for (size_type i=0; i<vector_size; ++i )
                    {
                        T a;
                        infile >> a;
                        temp_weight[i] = a;
                    }

                    it->second.weights[j] = temp_weight;
                }

                for (size_type j=0; j<it->second.bias.size(); ++j)
                {
                        T a;
                        infile >> a;
                        it->second.bias[j] = a;
                }




            }
            T a;
            infile >> a;  //dummy          
            if (!infile.eof())
                throw std::runtime_error("Not valid Weight format");
        }

        std::vector<T> predict (const std::vector <T> & input, int input_w, int input_h) 
        {
            std::vector <T> out_activation(input);
            for ( auto it= ++(layer_map.begin()); it!=layer_map.end(); ++it)
            {


                if (it->second.type_of_layer == layer_type::Conv2D)
                {
                    auto out_layer =  conv2d_helper_func(out_activation, input_w, input_h, it->second );
                    // perform activation if any! 
                    out_activation = impl_activation_func (out_layer, it->second.activation);

                }
                else if (it->second.type_of_layer == layer_type::depth_to_space)
                {
                    out_activation  = impl_depth_to_space(out_activation, input_w,  input_h, it->second.output_channel);
                }




            }

            return out_activation;
        }

        std::vector<T> impl_depth_to_space(const std::vector<T>& input, int input_w, int input_h, int input_c)
        {

            std::vector <T> out(input.size());
            int upscale_factor = std::sqrt(input_c);
            int j =-1;
            for (size_type i=0; i<input.size(); ++i)
            {

                if (i % (upscale_factor *input_w) ==0) // start of each row
                    ++j;
                out[i] = input[input_w * input_h * (j%upscale_factor)*upscale_factor  +  input_w * input_h * (i%upscale_factor)+ (i- j*input_w*upscale_factor)/upscale_factor+ i/(upscale_factor*input_w*upscale_factor)*input_w ];
            }
            return out;
        }



    private:
        int layer_number=1;
        std::map < std::string, layer_conf> layer_map;

        std::vector<T> impl_activation_func(const std::vector <T> & input, std::string activation_type)
        {
            std::vector<T> out(input);
            if (activation_type == "relu")
            {
                for (size_type i=0; i< input.size() ; i++)
                    out[i]=  (input[i]>0)?input[i]: 0;
            }


            else if (activation_type == "tanh")
            {

                for (size_type i=0; i< input.size() ; i++)
                    out[i]= std::tanh (input[i]);

            }

            return out;


        }

        std::vector<T> conv2d_helper_func(const std::vector <T> & input, int input_w, int input_h, const layer_conf &layer)
        {
            std::vector<T> out_t;
            for (size_type i=0; i<layer.weights.size(); ++i )
            {
                auto out_one_channel = impl_conv2d (input, layer.weights[i],  input_w, input_h, layer.input_channel, layer.stride, layer.padding, layer.filter_size, layer.bias[i]);
                out_t.insert(out_t.end(), out_one_channel.begin(), out_one_channel.end());
            }
            return out_t;
        }



        std::vector <T> impl_conv2d (const std::vector<T> & input, const std::vector<T> & weights,  int w, int h, int c, int stride, std::string padding, int filter_size, float bias=0)
        {
            if (padding == "same")
            {
                std::vector<T> out (input.size()/c);
                std::vector <T> in_padded ( (w +(filter_size-1)) * (h +(filter_size-1)) * c , 0 );
                size_type in_index = 0;
                for (size_type i = 0 ; i<in_padded.size(); ++i )
                {
                    size_type j =  i % (w +(filter_size-1)); // column index
                    size_type k = (i %  ((w +filter_size-1)   * (h +filter_size-1))) / (w +filter_size-1); // row index
                    size_type c_index = i /  ((w +filter_size-1)   * (h +filter_size-1)) ;    //channel index
                    if ( j >= (filter_size-1)/2 && k >= (filter_size-1)/2 && j < (w+(filter_size-1)/2) && k < (h+(filter_size-1)/2) )
                    {
                        in_padded[i] = input [in_index];
                        ++in_index;
                    }
                }


                size_type size_weight = weights.size(); //adding after valgrind
                size_type filter_size_2 = filter_size*filter_size;
                size_type padding_width = w +filter_size-1;
                for (size_type i = 0 ; i<h; ++i )
                    for (size_type j = 0 ; j<w; ++j)
                    {
                        size_type t=0; //for efficiency
                        for (size_type k =0; k< size_weight ; ++k)
                        {
                            t = k% ( filter_size_2 ) / filter_size * (padding_width) + k % filter_size + k / (filter_size_2)  * ((padding_width)   * (h +filter_size-1));
                            out[i *w + j] += weights[k] * in_padded [ i * (padding_width)   + j   + t  ];

                        }
                        out[i *w + j] += bias;

                    }


                
                return out;
            }
            
            else
            {
                return input;

            }
        }



    };
}