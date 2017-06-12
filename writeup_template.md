##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/output_1_1.png
[image2]: ./examples/output_1_2.png
[image3]: ./examples/output_3_1.png
[image4]: ./examples/output_3_3.png
[image5]: ./examples/output_12_0.png
[image6]: ./examples/output_12_2.png
[image7]: ./examples/output_12_3.png
[image8]: ./examples/frameno34HeatThresholded.jpg
[image9]: ./examples/frameno34Label.jpg
[image10]: ./examples/frameno34Img.jpg
[image11]: ./examples/frameno35HeatThresholded.jpg
[image12]: ./examples/frameno35Label.jpg
[image13]: ./examples/frameno35Img.jpg
[image14]: ./examples/frameno36HeatThresholded.jpg
[image15]: ./examples/frameno36Label.jpg
[image16]: ./examples/frameno36Img.jpg
[image17]: ./examples/frameno37HeatThresholded.jpg
[image18]: ./examples/frameno37Label.jpg
[image19]: ./examples/frameno37Img.jpg
[image20]: ./examples/frameno38HeatThresholded.jpg
[image21]: ./examples/frameno38Label.jpg
[image22]: ./examples/frameno38Img.jpg
[image23]: ./examples/frameno39HeatThresholded.jpg
[image24]: ./examples/frameno39Label.jpg
[image25]: ./examples/frameno39Img.jpg

[video1]: ./test_video_processed_2.mp4
[video2]: ./project_video_processed_2.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `extract_features` class with function named `computeHOG_features` in the 3rd code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

I finally settled for `RGB` color space as the car and non-car features can be differentiated in this color space. Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image3]
![alt text][image4]

As can be seen from the above images, all the the 3 channels HOG images differentiate a car and non car, further RGB color histogram is different between the 2 classes justifying the choice of `RGB` color space.

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the parameters mentioned in ###1 seems to work fine.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM and the code is given in code cell 7 of the Ipython notebook. The feature set used are spatial binning, HOG on all 3 channels and 3 channel histogram for the `RGB` image.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented using `find_cars` functions in code cell 35 of the Ipython notebook. In this function the HOG features are computed once for the scaled image and then corresponding features (HOG), spatial binning and color histogram are extracted/computed for different sliding windows. I have used a sliding step of 2 cells per step and the search is done for mutliple scales i.e. scale=1,2,4. This seems to work fine.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As mentioned above, I searched on three scales using RGB 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]
![alt text][image6]
![alt text][image7]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my test video result](./test_video_processed_2.mp4)
Here's a [link to my video result](./project_video_processed_2.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  Further, I maintained a list for last 5 heatmaps and averaged them before thresholding them to filter the flase detections. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. Further while constructing the heat map, I have added more heat to all the the bounding boxes wherein a positive detection is observed more than once. This helps in bounding boxes covering the entire car while false detections gets filtered. The code is given in `add_heat` function in code cell 35 of Ipython notebook.

Here's an example result showing the heatmap from a series of frames of test video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image8]

![alt text][image11]

![alt text][image14]

![alt text][image17]

![alt text][image20]

![alt text][image23]

### Here is the output of `scipy.ndimage.measurements.label()` on the thresholded averaged heatmap for all six frames:
![alt text][image9]

![alt text][image12]

![alt text][image15]

![alt text][image18]

![alt text][image21]

![alt text][image24]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image25]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

I have used `RGB` color space which works reasonably well although combined with other color space such as HSV (or a single channel in HSV which is more invariant to shadowing) might help to improve the false detection. Further some false detections are observed on the divider, one way is to filter them by constraining the region of interest (I havn't implemented it in the project).

Another important factor is to figure whether the pipeline can be run in real time or not. Tecnhiques to make it more efficient in terms of MIPS requirement is definitely valuable for real time deployment.

