# Multi-Number Detection

Environment : Python 3.X

### Required Dependencies
1. pandas
2. numpy
3. sklearn.ensemble
4. open-cv

### Description
This python program allows multi-number detection using Random Forest Classification
This is an application of applying Random Forest Classification for image detection
The script may need optimization in variables for better detection accuracy
Script tested on Python 3.6.4

### Usage
1. Run "mnist_pickle.py" to output "pickle.sav"
2. Run "detect_num.py"
3. Image with detection results should pop up on screen
3. If you would want to change the image, modify the following line
```python
img = cv2.imread("nums.png")
```

### Adjustable Variables
```python
thresh = 150 #set thresh for black and white sensitivity(higher for less sensitivity)
spc = 5 #spacing between each cropping
dist = 40 #detect overlap for same numbers
overlap = 10 # detect overlap between diff numbers
ratio = [(30,30),(40,40)] #ratio for cropping
```
