The Xuzhou dataset was collected by an airborne HYSPEX hyperspectral camera 
over the Xuzhou peri-urban site in November 2014. This dataset consists of 
500 × 260 pixels, with a very high spatial resolution of 0.73 m/pixel. 
The number of spectral bands used in the experiment was 436, after removing the noisy 
bands ranging from 415 nm to 2508 nm. The scene is peri-urban and is characterized by 
nine categories, including crops, vegetation, man-made structures, and coal fields 
The very high spatial resolution and the complex mixed categories make this dataset 
a challenging dataset for classification.
Instructions: 
The file declaration:
1.       “classification image.jpg”: The jpeg format of ground truth.
2.       “Pseudocolor image.jpg”: The jpeg format of pseudocolor.
3.       “image.hdr”/“image.img”: The raster data format of the Xuzhou HYSPEX dataset
4.       “true label.hdr”/“true label.img”: The raster data format of the ground truth
5.       “data.mat”: the matlab format of this dataset. The variable “all_x” and “all_y” 
represents the spectral vector and the corresponding label respectively of the whole pixels (13000). 
The unclassified pixels are labeled as 0 in “all_y”; The variable “x” and “y” represents 
the spectral vector and the corresponding label respectively of the all labeled pixels(68877).