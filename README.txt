# cpp_project-inference-espcn

C++ implementation of ESPCN algorithm described in [1]. This project was done for the final project of the "ECE 596C" course. The code is written from scratch and the only library used other than C++ STD is OpenCV. There is no limitation on the input size however the upscaling is used for small size images. For the 500*500 image the algorithm takes 2 minutes to produce the output image.

The "Nueral_net" template class is a general class to define CNNs in Keras software library manner. ESPCN was defined using the Nueral_net class. 

For running the code make do the following steps:
	1. go to the $TOP_DIR of the code, the $TOP_DIR denote the directory containing this README file.
	And let $INSTALL_DIR denote the directory into which this software is to be installed. 

	2. Go to the $TOP_DIR:
	    cd $TOP_DIR 

	3. To build and install the software, use the commands:
	    cmake -H. -Btmp_cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
	    cmake --build tmp_cmake --clean-first --target install *

* This command might need the root privilege

after running the above commands, an executable file will be generated in the $INSTALL_DIR called "upcale_image"

Now you can use the following command to upscale your images with the factor of Three! 

Go to the $INSTALL_DIR and run the followling command. 

`cat {path to jpg image} | ./upscale_image "path to network's weights" >  {upscaled_image.jpg} ` \
The above command read the input image from the standard input stream and write the image to the standard output stream which is stored in the "upscaled_image.jpg"`


References

[1] Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A., Bishop, R., Rueckert, D. and Wang, Z. 
(2016). Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
 Neural Network. Available at: https://arxiv.org/abs/1609.05158 \
