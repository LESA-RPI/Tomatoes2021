# Rensselaer Polytechnic Institute: Lighting Enabled Systems & Applications
### *Lighting Enabled Systems & Applications*

[![N|Solid](https://lesa.rpi.edu/wp-content/uploads/2021/04/PlantScienceTransIcon.jpg)](https://lesa.rpi.edu/index.php/research/horticulturallighting/)

[![Build](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://github.com/LESA-RPI/Tomatoes2021)

This repository automatically creates graphical data analysis for the Tomatoes in our Conviron Growth Chambers.

- This is the *ReadMe* file for *CHMOD’s Plant Database!*

- This is also an Open-Source github repository that aims to collaborate Rensselaer Polytechnic Institute’s Lighting and Systems Based Applications research with other Academic Institutions.

We are currently conducting research regarding the optimization of plant growth, and we are currently tracking CO2, Temperature, and Humidity levels to begin our experiment!

We are storing Metadata in regards to the sensors we are using right here:
## Metadata
#
#
Sensors and Cameras | Metadata Links
------------ | -------------
Raspberry Pi Camera | [RPI Metadata](https://static.raspberrypi.org/files/product-briefs/Raspberry_Pi_HQ_Camera_Product_Brief.pdf)
Raspberry Pi Lense | [Lense Metadata](https://cdn-shop.adafruit.com/product-files/4563/4563-datasheet.pdf)
Temperature and Humidity Sensor  |  [Sensor Link to Metadata](https://www.adafruit.com/product/4867)

#
#


![Set-up of Sensors](https://i.imgur.com/epB4Be8.jpg)
![Raspberry Pi Camera](https://i.imgur.com/aXhDS9o.jpg)
![Sensor Metadata](https://i.imgur.com/7441YCg.jpg)

## Tech
We are currently using the packages: Numpy, Pandas, Matplotlib, and Scipy. 
For the Image Analysis, we are looking towards CV2 and Detectron2, however this is a current work in progress.

## Run the code!

RPI's Tomato Repository requires [Python](https://phoenixnap.com/kb/how-to-install-python-3-ubuntu) to run. Nikita Bhagam uses Python3.


```sh
cd ./......./LESA-RPI/Tomatoes2021
python3 python3 Plant_Matrix.py
...enter start of night cycle in HH:MM:SS...
..enter end of night cycle in HH:MM:SS...
...enter name or date of last recording...
./tomatoes/Tomato_RPI_1/txt_files (interchangable with any tomato in repository)
```
A file parsing script is also included.
## License

*MIT License*
**Free to use as needed :D**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
