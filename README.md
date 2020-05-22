# Billed sort/zip projekt

## Authors
- **Gruppe:** happy county, 
    - Andreas Vikke https://andreasvikke.dk/
    - Martin Frederiksen https://github.com/MartinFrederiksen

## Description
I dette projekt vil vi gerne udarbejde en hjemmeside via flask, hvor det er muligt at uploade en zip fil.
Denne zip fil skal indeholde de ting, som står under brugsanvisningen.

Det skal så være muligt for vores hjemmeside at sortere denne zip fil vha. deep learning og convolutional neural network hvorefter mappen bliver zippet igen og bliver downloadet til din egen desktop.

Vi vil sortere på:
 - Deep learning, shape predictor ansigtsgenkendelse, kunne kende forskel på personer på billeder
 - Deep learning, CNN skal kunne genkende objekter som fx kat, hund, bil, fly.

## Video
 [![Billed sort/zip projekt](http://img.youtube.com/vi/LViM3LfONCE/0.jpg)](http://www.youtube.com/watch?v=LViM3LfONCE "Billed sort/zip projekt")

## Important external uses
|Libary|Version|
|---|---|
|[dlib](https://pypi.org/project/dlib/)|19.19.0|
|[opencv-python](https://pypi.org/project/opencv-python/)|4.2.0.34|
|[numpy](https://pypi.org/project/numpy/)|1.18.4|
|[tensorflow](https://pypi.org/project/tensorflow/)|2.2.0|
|[tqdm](https://pypi.org/project/tqdm/)|4.46.0|
|[imutils](https://pypi.org/project/imutils/)|0.5.3|
|[Flask](https://pypi.org/project/Flask/)|1.1.2|
|[Werkzeug](https://pypi.org/project/Werkzeug/)|1.0.1|

|Model|
|---|
|[dlib_face_recognition_resnet_model_v1](https://github.com/davisking/dlib-models)|
|[shape_predictor_68_face_landmarks](https://github.com/davisking/dlib-models)|
|[cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)|

## Teknologier brugt:
- Python
- Flask
- Numpy
- dlib
- OpenCV4
- Convolution Neural Network
- MySQL
- Zipfile
- DigitalOcean

## Udfordringer:
- Data collection
    - User input(zip file)
- Data wrangling
    - File names
    - Multiple folders called Train
    - Gif, png, jpg
- Data processing
    - Model fitting(image downscale)
    - Features i ansigt
- Presentation
    - Flask server

## Brugsanvisning:
#### Før brug:
1. Før hjemmesiden bliver brugt skal du oprette en mappe kaldet fx "Foo" med en undermappe kaldet "train". 
2. I mappen "train" ligger du de billeder af personer ind du gerne vil have trænet på.
3. Døb nu billederne i "train" mappen noget meningsfuldt altså fx MartinFrederiksen.png.
4. Lig så alle de billeder ind i "Foo" som du gerne vil have sorteret. Både personer og objekter
5. Nu kan du zippe "Foo" og uplaode den på Flask serveren.

#### Under brug:
1. Når filen er uploaded vil du inden længe kunne se alle billederne på hjemmesiden.
2. Her kan du sortere efter objekter og personer ved hjælp af knapperne
3. hvis du vil downloade den originale zip fil eller den sorterede zip fil kan du gøre det på knapperne.
