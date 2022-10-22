# SuecaAR

An augmented reality application for detecting and classifying playing cards. It enables the playing of the Portuguese Sueca card game.

The program has 2 different renderers: the OpenCV rendered (draws wireframes) and the OpenGL rendered which enables drawing 3d models.

## Setup

Install python 3.10.

### OpenCV renderer

Install the following packages:

```sh
pip install opencv-contrib-python numpy
```

### OpenGL renderer

For Windows:

```sh
pip install pillow pygame libs/PyOpenGL-3.1.6-cp310-cp310-win_amd64.whl libs/PyOpenGL_accelerate-3.1.6-cp310-cp310-win_amd64.whl
```

For linux:

```sh
pip install pillow pygame PyOpenGL PyOpenGL_accelerate
```

## Running the project

```sh
cd src
python play_game.py <camera_parameters(.npz format)> <camera_feed(int)> <number of game rounds> <trump_suit(Spades/Hearts/Diamonds/Clubs)>
```
