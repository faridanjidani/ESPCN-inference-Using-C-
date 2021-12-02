

#include <opencv2/core/core.hpp>           // cv::Mat
// #include <opencv2/imgcodecs/imgcodecs.hpp>     // cv::imread()
#include <opencv2/imgproc/imgproc.hpp> 
#include <iterator>
#include "ra/Neural_net.hpp"


#include "ra/imgcodecs.hpp"
namespace ra::networks

{

    class espcn 
    {
        public:


        // An unsigned integral type used to represent sizes.
        using size_type = std::size_t;


        espcn();
        espcn(const espcn&) = delete;
        espcn& operator=(const espcn&) = delete;
        espcn(espcn&&) = delete;
        espcn& operator=(espcn&&) = delete;


        ~espcn(){}

        void load_weights (std::string file);

        void read_input_image (std::istream& in);

        void upscal_image ();





        private:
        ra::neural_net::Neural_net <float> nn1;
        cv::Mat ycbr_split[3];
        std::vector<float>input_image;
        int input_height;
        int input_width;


    };



}