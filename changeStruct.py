import os
import shutil

# Path to the train folder
train_folder = './train/train'

# Path to the new train folder
new_train_folder = './changed_train'

# Create the new train folder if it doesn't exist
if not os.path.exists(new_train_folder):
    os.makedirs(new_train_folder)

# Iterate over the files in the train folder
for filename in os.listdir(train_folder):
    # Get the full path of the file
    file_path = os.path.join(train_folder, filename)
    
    # Check if the file contains 'cat' in its name
    if 'cat' in filename:
        # Create the cat folder if it doesn't exist
        cat_folder = os.path.join(new_train_folder, 'cat')
        if not os.path.exists(cat_folder):
            os.makedirs(cat_folder)
        
        # Move the file to the cat folder
        shutil.move(file_path, cat_folder)
    
    # Check if the file contains 'dog' in its name
    elif 'dog' in filename:
        # Create the dog folder if it doesn't exist
        dog_folder = os.path.join(new_train_folder, 'dog')
        if not os.path.exists(dog_folder):
            os.makedirs(dog_folder)
        
        # Move the file to the dog folder
        shutil.move(file_path, dog_folder)