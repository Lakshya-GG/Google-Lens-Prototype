```
model = EfficientNetB0(weights='imagenet', include_top=True)
```

### EfficientNetB0
- **EfficientNetB0** is a type of convolutional neural network (CNN) that belongs to the EfficientNet family. These networks are known for their efficiency in terms of accuracy and computational resource usage. The "B0" in the name indicates it's the base or smallest variant in the EfficientNet series, which ranges from B0 to B7 with increasing size and complexity.

### Parameters Explained
- **weights='imagenet'**: This parameter specifies that the model should be loaded with weights that have been pre-trained on the ImageNet dataset. ImageNet is a large visual database used for training deep neural networks in object recognition software. By specifying 'imagenet', you're loading weights that have been trained to recognize 1,000 different object categories found in the ImageNet competition.
  
- **include_top=True**: This parameter determines whether to include the fully connected layer at the top of the network. Setting `include_top=True` means the model will include the final fully connected layers, making it ready to classify images into 1,000 ImageNet classes directly. This is useful when you want to use the model in a standard image classification task with the same categories as the ImageNet dataset.


```
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("top_activation").output)
```

 create a feature extractor using a specific layer from a pre-trained model, in this case, the EfficientNetB0 model. Here’s what’s happening in the code and some insight into why the 'top_activation' layer might have been chosen for feature extraction.

### Understanding the Code

#### Model Subclassing to Extract Features
```python
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("top_activation").output)
```

- **Model(inputs, outputs)**: This syntax is used to create a new model in Keras. The `Model` class is instantiated with two primary arguments:
  - **inputs**: Specifies the input(s) of the model, which in this case is `model.input`. This refers to the input layer of the EfficientNetB0 model.
  - **outputs**: Defines what the output of the model will be. Instead of using the final output of the original model, you specify a different layer as the output.

- **model.get_layer("top_activation")**: This function retrieves a layer from the pre-trained model by its name. The layer’s output is then used as the output of the new model. This means that the new model (`feat_extractor`) will perform all the operations of the original model up through the 'top_activation' layer.

### Why Choose the 'top_activation' Layer?

- **High-Level Features**: In deep learning, layers closer to the output of the model tend to learn more abstract and high-level features. By using an output from a layer near the end of the model (but not the final output), you can capture complex features that are more informative about the content of the image than earlier layers. These features are generally better at distinguishing between different kinds of images with subtle differences.

- **Purpose of Feature Extraction**: Feature extraction models are often used in tasks where the goal is to compare images, search for similar images, or classify images into categories not covered by the original training dataset (a process known as transfer learning). The 'top_activation' layer is likely chosen because it provides a rich, yet compressed, representation of the image which can be very useful for these tasks.

- **Avoid Over-Specific Features**: The final layers of a network trained for classification (especially the last fully connected layers) are very specific to the classes on which the network was trained. By stopping before these layers, the 'top_activation' layer avoids capturing these overly specific features, which makes the extracted features more generalizable.

### Practical Use
In practical applications, the features extracted by this model can be used for:
- **Image Retrieval**: Finding images that are visually similar to a query image.
- **Clustering**: Grouping similar images together in an unsupervised manner.
- **Anomaly Detection**: Identifying images that do not conform to the pattern seen in the training dataset.

The choice of layer for feature extraction can significantly impact the performance of these tasks, depending on how relevant the learned features are to the task at hand. Choosing 'top_activation' is a strategy to balance between abstractness and specificity.


```

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize the image if necessary
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    return img, x

# Assuming you have defined feat_extractor and load_image functions

image_path = "/content/drive/MyDrive/AIWR/Dataset/download/Planes/4.jpeg"
img, x = load_image(image_path)
feat = feat_extractor.predict(x)
feat_flat = feat.flatten()

plt.figure(figsize=(16, 8))  # Adjust the figure size as needed

# Create the subplot for the plot of feat[0]
plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
plt.plot(feat_flat)
plt.title('Feature Plot')

# Create the subplot for displaying the image
plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
plt.imshow(img)
plt.title('Image')

plt.tight_layout()  # Ensure proper spacing between subplots
plt.show()

```



This Python script is designed to load an image, preprocess it, extract features using a pre-trained model, and then visualize both the extracted features and the image itself using Matplotlib. Here's a detailed breakdown of each part of the code:

### Import Libraries
```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
```
- **matplotlib.pyplot**: Used for creating plots and figures.
- **PIL.Image**: Part of the Python Imaging Library, used here to load and manipulate images.
- **numpy**: Essential for numerical operations on arrays.

### Define Function: `load_image`
```python
def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize the image if necessary
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    return img, x
```
- **Purpose**: To load an image from the specified path, resize it to 224x224 pixels (a common input size for convolutional neural networks), normalize the pixel values to the range [0, 1], and add a batch dimension to the image array, making it suitable for processing by neural network models which expect input in this format.
- **Returns**: The original PIL image object (`img`) and the processed image array (`x`).

### Load Image and Extract Features
```python
image_path = "/content/drive/MyDrive/AIWR/Dataset/download/Planes/4.jpeg"
img, x = load_image(image_path)
feat = feat_extractor.predict(x)
feat_flat = feat.flatten()
```
- **Load and Process Image**: The image at the specified path is loaded and processed using the `load_image` function.
- **Feature Extraction**: The pre-trained model `feat_extractor` is used to predict (extract) features from the processed image. These features (`feat`) are then flattened into a one-dimensional array (`feat_flat`) for easier visualization.

### Visualization Using Matplotlib
```python
plt.figure(figsize=(16, 8))  # Adjust the figure size as needed

# Create the subplot for the plot of feat[0]
plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
plt.plot(feat_flat)
plt.title('Feature Plot')

# Create the subplot for displaying the image
plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
plt.imshow(img)
plt.title('Image')

plt.tight_layout()  # Ensure proper spacing between subplots
plt.show()
```
- **Figure Setup**: A figure is created with a size of 16x8 inches.
- **Feature Plot**: The first subplot (1x2 grid, position 1) displays a plot of the flattened feature vector. This plot helps visualize the pattern of activations extracted by the model, which can be indicative of how the model is interpreting different aspects of the image.
- **Image Display**: The second subplot (1x2 grid, position 2) shows the original image, allowing for visual comparison between the image and its feature representation.
- **Layout and Display**: `plt.tight_layout()` improves the spacing between subplots for clarity, and `plt.show()` renders the figure on the screen.

This script is a useful tool for understanding how different images are represented internally by the features extracted from a deep learning model. It combines image processing, feature extraction, and visualization in a concise workflow.


# Creating Postings List

Python dictionary comprehensions used to create two dictionaries: `dict` and `feature_dict`. Each dictionary is initialized with keys ranging from 0 to 79, and each key is associated with an empty list as its value. Here's a detailed explanation of each part:

### Dictionary Comprehension
The syntax used here is a form of dictionary comprehension, which is a concise way to create dictionaries in Python. The structure of a dictionary comprehension is similar to that of a list comprehension, but it constructs a dictionary instead of a list.

#### Syntax
```python
{key_expression: value_expression for variable in iterable}
```
- **key_expression**: This defines what the keys will be. In your case, it's simply `i`, which represents each integer from 0 to 79.
- **value_expression**: This defines the values associated with each key. Here, it’s `[]`, an empty list.
- **for variable in iterable**: This loops over some iterable object, assigning each value it contains to `variable`, one at a time. Here, `range(0, 80)` generates numbers from 0 to 79.

### Specific Examples

#### 1. `dict = {i: [] for i in range(0, 80)}`
This dictionary comprehension creates a dictionary named `dict` where each key is an integer from 0 to 79, and each value is an empty list. This kind of structure might be used in situations where you need to categorize or store items under numbered categories that initially have no members. It's important to note that `dict` is a built-in type in Python, so using it as a variable name can overshadow the built-in `dict` type, which is generally not advisable.

#### 2. `feature_dict = {i: [] for i in range(0, 80)}`
Similarly, this dictionary comprehension creates a dictionary named `feature_dict` with the same structure as `dict`. Each key (0 to 79) is associated with an empty list. The naming suggests that this dictionary is intended to store features or data related to features, where each key might represent a specific feature type or category.


# Import Model

```

from ultralytics import YOLO
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
object_detection_model = YOLO('yolov8n.pt')
```



# Posting List for a object || Creating Posting List for a single object



```
image_path = '/content/drive/MyDrive/AIWR/Dataset/download/Fruits/31.jpeg'
results = object_detection_model( image_path, conf = 0.5)

# extracting results
image_results = results[0].boxes.cpu().numpy()

# getting class labels
class_labels = image_results.cls
class_labels = set(class_labels)

# adding image in respective class label
for i in class_labels:
    dict[i].append(image_path)

results[0].names

```

 involves loading an image, performing object detection on it, extracting results, and organizing these results. Here's a detailed breakdown and explanation of each step and how it all ties together:

### 1. **Load Image Path**
```python
image_path = '/content/drive/MyDrive/AIWR/Dataset/download/Fruits/31.jpeg'
```
This line defines the path to an image file that you want to process. This path is used to load the image into the object detection model.

### 2. **Object Detection**
```python
results = object_detection_model(image_path, conf = 0.5)
```
- **object_detection_model**: This is presumably a pre-loaded model capable of detecting objects in images. You pass the image path directly to the model.
- **conf = 0.5**: This parameter sets the confidence threshold at 0.5. The model will only return detections that it is at least 50% confident about.

### 3. **Extract Detection Results**
```python
image_results = results[0].boxes.cpu().numpy()
```
- **results[0].boxes**: After detecting objects, the model returns results, typically in a batched format where each entry in the batch corresponds to a different aspect of the detection (like bounding boxes, scores, etc.). Here, `results[0].boxes` likely refers to the bounding boxes of detected objects in the first item of the results, which corresponds to your image.
- **.cpu().numpy()**: This sequence of calls is common in PyTorch (assuming the use of PyTorch due to syntax), where `.cpu()` moves tensors to the CPU from the GPU (if they were on the GPU), and `.numpy()` converts these tensors to NumPy arrays for easier manipulation in Python.

### 4. **Extract Class Labels**
```python
class_labels = image_results.cls
class_labels = set(class_labels)
```
- **image_results.cls**: This line suggests that the `image_results` array has an attribute or field named `cls`, which contains the class labels of the detected objects.
- **set(class_labels)**: Converts the list of class labels into a set, which automatically removes any duplicates, ensuring each class label is unique.

### 5. **Store Image Path in Corresponding Class Label**
```python
for i in class_labels:
    dict[i].append(image_path)
```
- **dict[i].append(image_path)**: This loop iterates over each unique class label detected in the image and appends the image path to a dictionary under the corresponding class label. This is where the dictionary `dict` that you initialized earlier (with keys 0 to 79 and empty lists as values) is used. Each class label indexes into the dictionary, and the image path is added to the list corresponding to that label.

### 6. **Access Detection Names**
```python
results[0].names
```
- **results[0].names**: This part of the code is likely retrieving the names (or descriptions) of the detected classes. It’s not used directly in the snippet you've provided but could be useful for debugging or logging to understand what kinds of objects the model is detecting in the image.

### Summary
The overall process involves detecting objects in a given image, extracting their class labels, and organizing the image paths by these labels in a dictionary. This can be particularly useful in applications where you need to catalog images by detected features or where multiple images need to be processed and categorized based on the content detected within them.

# Create Posting lists for objects and all images in dataset


# CodeBuddy


```
import ReadBuddy as rb
my_reader=rb.ReadBuddy("/content/drive/MyDrive/AIWR/Dataset/download")
_,folder_dict=my_reader.create_folder_dictionary()
```

```
def get_class_id(model, image_path):
    results = model( image_path, conf = 0.5)

    # extracting results
    image_results = results[0].boxes.cpu().numpy()

    # getting class labels
    class_labels = image_results.cls
    class_labels = set(class_labels)
    return class_labels
```


```
def update_posting_list(model, image_path , dict ):

    class_labels = get_class_id(model, image_path)
    # adding image in respective class label
    for i in class_labels:
        dict[i].append(image_path)
    return dict
```


```

for i in folder_dict:

    for image_path in folder_dict[i]:
        update_posting_list(object_detection_model, image_path, dict)



```

 processing an image dataset and organizing it based on the objects detected in each image. This organization is facilitated through a data structure known as a "posting list." Let's break down each component of the script to understand its function and flow:

### Libraries and Initial Setup
```python
import ReadBuddy as rb
```
- **ReadBuddy**: This appears to be a custom or third-party library (not standard in Python), which is used to read and manage files, likely specialized for handling datasets stored in directories.

```python
my_reader = rb.ReadBuddy("/content/drive/MyDrive/AIWR/Dataset/download")
_, folder_dict = my_reader.create_folder_dictionary()
```
- **my_reader**: An instance of `ReadBuddy`, initialized with the path to your dataset. This object is likely responsible for accessing and managing the data in your specified directory.
- **create_folder_dictionary()**: A method that probably scans the given directory and returns a dictionary where keys are folder names and values are lists of image paths within those folders. The underscore `_` is used to ignore the first return value, implying that `create_folder_dictionary()` might return multiple pieces of data, of which only the folder-to-image mappings are needed.

### Functions Defined
#### 1. `get_class_id`
This function performs object detection on a given image and extracts unique class labels of the detected objects.
```python
def get_class_id(model, image_path):
    results = model(image_path, conf=0.5)
    image_results = results[0].boxes.cpu().numpy()
    class_labels = image_results.cls
    class_labels = set(class_labels)
    return class_labels
```
- **model(image_path, conf=0.5)**: Executes the object detection model on the specified image with a confidence threshold of 0.5.
- **results[0].boxes.cpu().numpy()**: Extracts the bounding boxes of detected objects and converts them into a NumPy array.
- **image_results.cls**: Presumably retrieves the class labels associated with the detected bounding boxes.
- **set(class_labels)**: Removes duplicate labels by converting the list of class labels into a set.

#### 2. `update_posting_list`
This function updates the posting list with the path of an image under each class label detected in that image.
```python
def update_posting_list(model, image_path, dict):
    class_labels = get_class_id(model, image_path)
    for i in class_labels:
        dict[i].append(image_path)
    return dict
```
- **get_class_id(model, image_path)**: Calls the previously defined function to get detected class labels.
- **for i in class_labels**: For each unique class label, append the current image path to the corresponding list in the dictionary `dict`.

### Iterating Through the Dataset
```python
for i in folder_dict:
    for image_path in folder_dict[i]:
        update_posting_list(object_detection_model, image_path, dict)
```
- **for i in folder_dict**: Iterates over each key in `folder_dict`, where each key represents a folder containing images.
- **for image_path in folder_dict[i]**: Iterates over each image path listed under the current folder.
- **update_posting_list(object_detection_model, image_path, dict)**: Updates the posting list with the current image, categorized by the objects detected in the image.

### Summary
 to categorize images based on detected objects, storing these categorizations in a "posting list" which is essentially a dictionary where each key is an object class, and the value is a list of image paths that contain that object. This structure is particularly useful for tasks like image retrieval, where you might want to find all images containing a specific type of object. The use of a custom library (`ReadBuddy`) suggests that the dataset management is tailored to specific requirements or dataset structures.


# Extract features and put them in new feature posting list

# Using VGG16 model to extract features as it performed better

setting up a process to extract features from images using the VGG16 model, a popular deep learning model for image classification and feature extraction. Here's a detailed breakdown of each part of the code:

### Import Libraries
```python
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
```
- **keras**: A high-level neural networks API, capable of running on top of TensorFlow, CNTK, or Theano.
- **image**: Module from Keras for image preprocessing.
- **decode_predictions, preprocess_input**: Utilities for processing the outputs of the model and preparing image data for prediction.
- **Model**: Used to instantiate a new model object from an existing model.
- **matplotlib.pyplot**: For plotting images (not used in the code shown but useful for visualization).
- **PIL.Image**: For image file manipulation.
- **numpy**: For numerical operations on arrays.

### VGG16 Model Setup
```python
model = keras.applications.VGG16(weights='imagenet', include_top=True)
```
- **VGG16**: This is a convolutional neural network model that is very deep and has 16 layers. It's widely used for image recognition tasks.
- **weights='imagenet'**: Indicates that the model should be loaded with weights pre-trained on the ImageNet dataset, which contains a large set of diverse images used for image classification.
- **include_top=True**: This includes the fully connected layers at the top of the network, which are typically used for classifying the images into one of the 1,000 classes of ImageNet.

### Feature Extractor Setup
```python
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
```
- **feat_extractor**: This line creates a new model that will use the input from the original VGG16 model but will output from an intermediate layer called "fc2". The "fc2" layer is one of the last fully connected layers in VGG16, just before the final classification layer. This makes it rich in the abstract features that the network has learned but not yet converted into final class predictions. It’s great for tasks that need general features but not specific class labels.

### Load Image Function
```python
def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize the image if necessary
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    return img, x
```
- **load_image**: A function to load and preprocess an image so it can be fed into the VGG16 model.
- **Image.open(image_path)**: Opens an image file.
- **img.resize((224, 224))**: Resizes the image to 224x224 pixels, the input size expected by VGG16.
- **np.array(img) / 255.0**: Converts the image to a numpy array and normalizes its pixel values to be between 0 and 1.
- **np.expand_dims(x, axis=0)**: Adds a batch dimension to the array, converting it from shape `(224, 224, 3)` to `(1, 224, 224, 3)`, which is necessary because Keras models expect a batch of images as input, even if you are only processing one image.

### Summary
This setup is ideal for extracting deep features from images using a pre-trained VGG16 model, which can then be used for various applications such as image retrieval, classification with different datasets, or further image analysis tasks. The extracted features are high-level representations learned by the model when trained on a vast array of ImageNet images, making them robust and versatile for many vision-based tasks.

# Creating Feature Posting List for detected objects in the dataset


```
for i in dict:
    for image_path in dict[i]:
        img, x = load_image(image_path)
        feat = feat_extractor.predict(x)[0]
        feature_dict[i].append(feat)

```

# Now at this point feature extraction is done all i have to do is

* get a new image
* run the object detection routine
* get all the feature vectors for the detected objects from feature posting list


* extract features for the new image
* do a cosine similarity with the small feature posting list

* return results

describes a comprehensive image processing workflow where an image is loaded, its features are extracted using a pre-trained model (VGG16), and similar images are retrieved based on cosine similarity of feature vectors. The code also involves object detection to refine the search space. Let's break down and explain each segment of the script:

### 1. **Loading and Displaying Features of an Image**
This part of the script focuses on loading an image, extracting its features using the VGG16 model, and plotting these features along with displaying the image itself.

```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

image_path = "/content/drive/MyDrive/AIWR/Dataset/download/Dog/17.jpeg"
img, x = load_image(image_path)
feat = feat_extractor.predict(x)

plt.figure(figsize=(16, 8))
plt.subplot(3, 2, 1)
plt.plot(feat[0])
plt.title('Feature Plot')

plt.subplot(2, 2, 2)
plt.imshow(img)
plt.title('Image')

plt.tight_layout()
plt.show()
```
- **load_image**: Loads and preprocesses the image to the required format for the model.
- **feat_extractor.predict**: Extracts features from the preprocessed image.
- **Plotting**: Uses Matplotlib to create subplots showing both the feature vector and the image itself.

### 2. **Iterate Over a Feature Dictionary**
This part attempts to visualize features for each image classified under a specific class (class 16 here) from a dictionary where features have been stored. However, the code snippet appears to be incomplete or incorrectly explained, especially the parts involving plotting these feature vectors.

### 3. **Feature Extraction and Retrieval Based on Cosine Similarity**
This segment of the code handles the feature extraction of a new query image, follows object detection to determine relevant classes, and uses cosine similarity to find and display the most similar images from a predefined feature space.

```python
image_path = "/content/drive/MyDrive/AIWR/Dataset/download/Dog/17.jpeg"
new_image, x = load_image(image_path)
new_features = feat_extractor.predict(x)

class_id = get_class_id(object_detection_model, image_path)
search_space = []
for i in class_id:
    search_space += feature_dict[i]  # Assuming feature_dict[i] is iterable

from scipy.spatial import distance
n_distances = [distance.cosine(new_features[0], feat) for feat in search_space]
n_idx_closest = sorted(range(len(n_distances)), key=lambda k: n_distances[k])[0:5]
n_results_image = get_concatenated_images(n_idx_closest, 200)

plt.figure(figsize=(5,15))
plt.imshow(new_image)
plt.title("query image")

plt.figure(figsize=(16,8))
plt.imshow(n_results_image)
plt.title("Retrieved Results")
```
- **Object Detection**: Detects objects in the new image and retrieves relevant class IDs.
- **Feature Extraction**: For the query image and computes the cosine distances to features in the selected class space.
- **Displaying Results**: Shows the query image and the top 5 similar images based on the smallest cosine distances.

### 4. **Function Definitions and Usage**
- **get_class_id**: Identifies and returns class IDs based on detected objects in the image.
- **get_concatenated_images**: A utility to concatenate image thumbnails for display based on indices of closest matches.

### Summary
Overall, the script integrates image loading, feature extraction, object detection, and similarity comparison into a cohesive image retrieval system. It uses advanced deep learning models for feature extraction and object detection to ensure that the retrieval process is both accurate and relevant. The visualizations provide a way to manually inspect the quality of the feature extractions and the final retrieved results. This kind of system is typical in scenarios like digital asset management, where similar images need to be quickly retrieved from large databases.




# Image Retrieval using Object Detection and Feature Extraction

 outlines a Python function for retrieving similar images based on object detection and feature extraction techniques. This is particularly useful in applications like image search engines, content-based image retrieval systems, and digital libraries. Let's break down the process and explain each segment in detail:

### Function: get_concatenated_images_from_reduced_dictionary
This function generates a concatenated image from a subset of images that are identified as being similar to a query image.

```python
def get_concatenated_images_from_reduced_dictionary(indexes, thumb_height, image_dict):
    thumbs = []
    for idx in indexes:
        img = image.load_img(image_dict[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image
```
- **Parameters**:
  - **indexes**: Indices of the images that are closest to the query image.
  - **thumb_height**: The height to which all images will be resized, maintaining aspect ratio.
  - **image_dict**: A dictionary where keys are indices and values are paths to images.
- **Process**:
  - For each index in `indexes`, load the image, resize it based on the specified thumbnail height, and append it to a list.
  - Concatenate all the thumbnail images horizontally to form a single image strip.
- **Return**:
  - The concatenated image which can be displayed to show the most similar images to the query.

### Function: retriveImage
This function orchestrates the entire process of loading a query image, extracting its features, comparing these features against a database using cosine similarity, and displaying the results.

```python
def retriveImage(image_path):
    new_image, x = load_image(image_path)
    new_features = feat_extractor.predict(x)

    # get class ids of image
    class_id = get_class_id(object_detection_model, image_path)

    reduced_feature_space = []
    reduced_image_space = []
    for i in class_id:
        reduced_feature_space += feature_dict[i]
        reduced_image_space += dict[i]

    n_distances = [distance.cosine(new_features[0], feat) for feat in reduced_feature_space]
    n_idx_closest = sorted(range(len(n_distances)), key=lambda k: n_distances[k])[0:5]
    n_results_image = get_concatenated_images_from_reduced_dictionary(n_idx_closest, 200, reduced_image_space)

    # display the results
    plt.figure(figsize=(5,15))
    plt.imshow(new_image)
    plt.title("query image")

    plt.figure(figsize=(16,8))
    plt.imshow(n_results_image)
    plt.title("Retrieved Results")
```
- **Image Loading and Feature Extraction**:
  - Load the query image and preprocess it for feature extraction.
  - Use a pre-trained model (like VGG16) to predict the features of the query image.
- **Object Detection**:
  - Run object detection on the query image to get class IDs which will be used to filter the search space.
- **Creating Reduced Feature Space**:
  - Filter the feature vectors and image paths based on the detected class IDs, creating a smaller search space that is more relevant to the content of the query image.
- **Cosine Similarity Calculation**:
  - Compute the cosine similarity between the feature vector of the query image and each feature vector in the reduced feature space.
  - Sort the results and select the indices of the closest images.
- **Image Display**:
  - Use `get_concatenated_images_from_reduced_dictionary` to create a single image from the closest matches and display it.
  - Show both the query image and the retrieved results to provide a visual comparison.

### Execution
Finally, the function is called with a specific image path, which triggers the retrieval process using the defined functions.

This method is effective in narrowing down search results by using object detection to ensure that the comparison is only performed against similar types of images, increasing both the efficiency and accuracy of the image retrieval system.
