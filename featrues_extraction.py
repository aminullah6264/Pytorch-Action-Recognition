import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
# Define the file path to your dataset
data_dir = './YouTube_DataSet_Annotated/action_youtube_naudio'

# Define the number of frames to extract features from
num_frames = 15

# Define the pre-processing steps for the images
# Define the pre-processing steps for the images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the pre-trained ResNet50 model
resnet = models.resnet18(pretrained=True).cuda()

# Remove the last layer of the ResNet50 model to obtain the feature extractor
resnet_feat = torch.nn.Sequential(*list(resnet.children())[:-1])
# Create an empty list to store the features and labels
samples = []
resnet.eval()
# Loop over the videos in the dataset folder
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    print(label_dir)
    for sub_dir in os.listdir(label_dir):
        if sub_dir == 'Annotation':
            continue
        video_dir = os.path.join(label_dir, sub_dir)
        for video_file in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = transform(frame).cuda()
                    frames.append(frame)
                    if frame_count == num_frames:
                        break
                else:
                    break
            cap.release()
            if len(frames) == num_frames:
                # Stack the frames into a tensor of shape (num_frames, 3, 224, 224)
                frames_tensor = torch.stack(frames, dim=0)
                # Extract the features using the pre-trained ResNet50 model
                with torch.no_grad():
                    features_tensor = resnet_feat(frames_tensor)
                    # print(features_tensor.shape)
                # Flatten the features tensor
                features_tensor = torch.flatten(features_tensor, start_dim=1)
                # Convert the features tensor to a numpy array
                features = features_tensor.cpu().numpy()
                # Append the features and label to the samples list
                samples.append((features, label))

# Shuffle the samples list
np.random.shuffle(samples)

# Split the samples into training and testing sets
split_idx = int(0.8 * len(samples))
train_samples = samples[:split_idx]
test_samples = samples[split_idx:]

# Separate the features and labels into separate arrays for training and testing sets
train_features, train_labels = zip(*train_samples)
test_features, test_labels = zip(*test_samples)

# Convert the labels to numerical labels using a LabelEncoder
le = LabelEncoder()
train_numerical_labels = le.fit_transform(train_labels)
test_numerical_labels = le.fit_transform(test_labels)
# Convert the features and labels arrays to numpy arrays
# Convert the features and labels arrays to numpy arrays
train_features = np.array(train_features)
train_labels = train_numerical_labels
test_features = np.array(test_features)
test_labels = test_numerical_labels

# Print the shapes of the features and labels arrays
print("Train Features shape:", train_features.shape)
print("Train Labels shape:", train_labels.shape)
print("Test Features shape:", test_features.shape)
print("Test Labels shape:", test_labels.shape)


# Save the features and labels to numpy arrays
np.save('train_features.npy', train_features)
np.save('train_labels.npy', train_labels)
np.save('test_features.npy', test_features)
np.save('test_labels.npy', test_labels)