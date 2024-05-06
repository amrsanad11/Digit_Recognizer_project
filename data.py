from tensorflow import keras

def data(train_path, test_path):
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=15,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
    )
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Load and preprocess training data using flow_from_directory
    train_ds = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(64, 64),
    batch_size=32,

    )

# Load and preprocess test data using flow_from_directory
    test_ds = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=(64, 64),
    batch_size=32,

    )

# Print the shapes of the first batch in each dataset
    # print("Train data batch shape:", next(train_ds).shape)
    # print("Test data batch shape:", next(test_ds).shape)
    return train_ds,test_ds