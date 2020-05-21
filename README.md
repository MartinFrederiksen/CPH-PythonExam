## Billed sort/zip projekt
**Gruppe:** happy county, **Medlemmer:** Andreas Vikke, Martin Frederiksen


I dette projekt vil vi gerne udarbejde en hjemmeside via flask, hvor det er muligt at uploade en zip fil.
Denne zip fil skal indeholde de ting, som står under brugsanvisningen.

Det skal så være muligt for vores hjemmeside at sortere denne zip fil vha. deep learning og convolutional neural network hvorefter mappen bliver zippet igen og bliver downloadet til din egen desktop.

***Hvis vi har mere tid:***
 - Deep learning skal kunne genkende objekter som fx kat, hund, bil, fly.
 - Mulighed for billedopbevaring på vores flask server


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
    - Mysql(Hvis mere tid)
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
1. Før hjemmesiden bliver brugt skal du oprette en mappe kaldet fx "Foo" med en undermappe kaldet "Train". 
2. I mappen "Train" ligger du de billeder ind du gerne vil have trænet på.
3. Døb nu billederne i "Train" mappen noget meningsfuldt altså fx MartinFrederiksen.png.
4. Lig så alle de billeder ind i "Foo" som du gerne vil have sorteret.
5. Nu kan du zippe "Foo".

#### Under brug:
TBD - Brugsanvisning for hjemmeside.
