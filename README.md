# Image Classifier using VGG16

Considering resource availability, I have taken a subset of the provided data. I have considered 7 categories from the provided data. 
Here are the following classes I have chosen: capsule_crack, cadpluse_faulty_imprint, capsule_good, hazelnut_crack, 
hazelnut_cut, hazelnut_good, hazelnut_hole.

I have used VGG16 to classify the data. Unfortunately, I spend very little time on the problem. I only got a chance to train for 2 epochs 
and my accuracy was 37% on test data. The size of my saved model is greater than 180MB and I couldn't add it to github. 

The `inference.py` scripts takes the model and test image and generates the prediction.

