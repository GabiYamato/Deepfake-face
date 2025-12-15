# Deepfake Face Detection

## Dataset Visualization

Sample images from the dataset (Train & TEST):

![Train Images](images/train_samples.png)
![Val Images](images/val_samples.png)

Sample Transformed Images (before model predicts):

![Sample Images](images/sample_images.png)

The dataset contains:
- **Real faces**: 5,890 images
- **Fake faces**: 7,000 images
- **Total**: 12,890 images

Images are organized in a standard ImageFolder structure with `Real` and `Fake` subdirectories.



## Model Architecture

![model resnet](images/resnet.jpg)

The model is a custom ResNet-inspired CNN trained from scratch with the following components:

### Architecture Details
- **Base**: ResNet-style residual blocks
- **Input**: RGB images (224x224x3)
- **Convolutional Layers**: Multiple residual blocks with skip connections
- **Activation**: ReLU activation functions
- **Pooling**: MaxPooling and Average Pooling layers
- **Fully Connected Layers**: Dense layers for binary classification
- **Output**: Single neuron with Sigmoid activation (Real vs Fake)

### Key Features
- Residual connections to prevent vanishing gradients
- Batch normalization for stable training
- Dropout layers for regularization
- Binary cross-entropy loss function
- Adam optimizer



## Training History

![training history](images/training_history.png)

## Test Confusion matrix

![Test Confusion Matrix](images/confusion_matrix.png)


## premature test results:

Test Accuracy: 0.7347
Test Precision: 0.7305
Test Recall: 0.6670
Test F1-Score: 0.6973
