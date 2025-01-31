##defining expected convolutional neural network model implementation

def create_custom_model(size=256, dropout_rate=0.5, classes=1):
    model = Sequential()
    # Block 1
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001), input_shape=(size, size, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))  # Early dropout

    # Block 2
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    # Block 3
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.4))

    # Block 4
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.4))

    # Block 5
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.5))  # Strongest dropout here

    # Global Average Pooling for feature reduction
    model.add(GlobalAveragePooling2D())

    # Fully Connected Layers
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Output Layer (Binary classification - Malignant vs. Benign)
    model.add(Dense(classes, activation='sigmoid'))  # Sigmoid for binary classification

    return model
