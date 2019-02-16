# vCanteen-crowd-estimator

An **unofficial** implementation of CVPR2016 paper [Single-Image Crowd Counting via Multi-Column Convolutional Neural Network](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)

This code use the pre-trained weight from this [github](https://github.com/uestcchicken/crowd-counting-MCNN)

We use Keras as an implementation **ONLY**

## Installation
1. Install Keras. 
```sh
pip install keras
```
or
```sh
pip3 install keras
```
2. Install Jupyter. 
```sh
pip install jupyter
```
3. Clone this repository. 
```
git clone https://github.com/tann9949/vCanteen-crowd-estimator.git
```

## Predicting headcount with your images
1. Launch jupyter notebook and open `Crowd Count MCNN_icanteen.ipynb`.
2. **Change the `img_path` of every cell to be the PATH to your images.**
3. **Change the `name` of the loaded image (see the line with `cv2.imread`).**
4. Enjoy estimating the crowd.

## Label your own crowd dataset
1. Launch `image_preprocessor/Head_Labeler.m` with Matlab.
2. Change `num_images`, `img_path` and `img_name` to match with your dataset.
3. Run `Head_Labeler.m`
4. Mark the head on your images by clicking on the head (one point per head is enough).
5. To exit, close the figure.

### Note for labeling with `getpts`
1. To delete the latest label, press `backspace`.
2. To finish labeling, press `return`.

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
Actual = 146<br>
<img src="/icanteen_img/test_5.jpg" width="350">

**OUTPUT**<br>
Prediction = 158<br>
<img src="/icanteen_heat/heat_test_5.png" width="350">

