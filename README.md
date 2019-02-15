# vCanteen-crowd-estimator

An **unofficial** implementation of CVPR2016 paper [Single-Image Crowd Counting via Multi-Column Convolutional Neural Network](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)

This code use the pre-trained weight from this [github](https://github.com/uestcchicken/crowd-counting-MCNN)

We use Keras as an implementation **ONLY**

## Installation
1. Install Keras. `pip install keras`  or `pip3 install keras`.
2. Install Jupyter. `pip install jupyter`
3. Clone this repository. `git clone https://github.com/tann9949/vCanteen-crowd-estimator.git`

## Predicting headcount with your images
1. Launch jupyter notebook and open `Crowd Count MCNN_icanteen.ipynb`.
2. **Change the `img_path` of every cell to be the PATH to your images.**
3. **Change the `name` of the loaded image (see the line with `cv2.imread`).**
4. Enjoy estimating the crowd.

## Other note
It is recommended to read the paper before try using this code to guarantee an understanding of the topics.
Prerequisites include: 
- Neural network.
- Convolutional Neural Network.
- Keras.
- Python Programming.

## Example
The input file and the output density map is as following:

**INPUT**<br>
Actual = 146
<img src="/icanteen_img/test_5.jpg" width="350">

**OUTPUT**<br>
Prediction = 158
<img src="/icanteen_heat/heat_test_5.png" width="350">

