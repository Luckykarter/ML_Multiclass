Main script: multiclass_detect

This project includes and extends ML_BinaryNetwork repository for any number of objects.
When two objects given - it uses binary model, else - categorical model.

Categories are defined from the respective folders.
E.g. to train model for horses, cats, dog - train and validation directory must contain the following sub-directories:
- horse
- cat
- dog

All dialogs with the user are done through pop-ups to load/save files

The workflow of the script:
1. Open pop-up window to choose folder with training data (and optionally validation data)
2. If there is existing trained Keras model - it can be loaded and used
3. If model is not chosen - the new model of CNN will be created and trained using provided images
4. After neural network is trained - it will be asked to save the Keras model (*.h5) 
5. User will be continuosly asked to upload an image to recognize - until they click "Cancel"
6. Give the result of what is on picture with percentage


**Example of resources:**

Training pictures for horses:

 https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip
 
 Validation pictures for horses:
 
  https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip
  
The archive has to be unpacked and chosen in the script