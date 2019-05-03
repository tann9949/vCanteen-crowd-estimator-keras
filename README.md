# vCanteen-crowd-estimator

An **unofficial** implementation of CVPR2016 paper [Single-Image Crowd Counting via Multi-Column Convolutional Neural Network](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)

My source for the files `train_preprocessing.m`, `get_density_map_gaussian.m`  and `weight.h5` are from [uestcchicken](https://github.com/uestcchicken). This is the link to his [github](https://github.com/uestcchicken/crowd-counting-MCNN) about the implementation of this paper too.

I wholeheartly thank him for his contribution. Without him(or her) this project wouldn't be complete.

We use Keras as an implementation **ONLY**

## Installation
1. Install Keras, Tensorflow. 
```sh
pip3 install keras
pip3 install tensorflow
```
2. Install Jupyter. 
```sh
pip3 install jupyter
```
3. Clone this repository. 
```
git clone https://github.com/tann9949/vCanteen-crowd-estimator.git
```

## To launch it on your camera
1. In `vCanteen.py`, line 141, delete argument `videopath`.
2. Run this command on your terminal/command prompt
```
python3 vcanteen.py
```

## To launch it on your video file
1. Add your video to `icanteen_video` directory.
2. In `vCanteen.py`, change the `videopath` variable (line 140) as your video.
3. Run this command on your terminal/command prompt
```
python3 vcanteen.py
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

## Authors
- **Chompakorn Chaksangchaichot** (5931229821)
- **Peeramit Masana** (5931316721)
- **Akekamon Boonsith** (5931393021)
