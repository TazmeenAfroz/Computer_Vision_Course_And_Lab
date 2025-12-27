"""
=================================================================
CNN EXAM CODE REFERENCE - ALL MINIMAL STRUCTURES
=================================================================
Roll: 22P-9252 | Prepared from Labs 8, 9, 10
Run on CPU | TensorFlow/Keras

TABLE OF CONTENTS:
1. Complete Base CNN Pipeline (MNIST)
2. AlexNet Structure
3. VGG16 Structure  
4. VGG19 Structure
5. Inception Block & Mini GoogLeNet
6. ResNet Blocks (Basic + Bottleneck)
7. Pretrained Model Usage
8. Image Loading Methods
9. U-Net for Segmentation
=================================================================
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, AveragePooling2D,
    BatchNormalization, Dropout, Flatten, Dense, 
    Activation, Add, Concatenate, GlobalAveragePooling2D,
    Conv2DTranspose, concatenate, Rescaling
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("TensorFlow Version:", tf.__version__)
print("=" * 60)

# =================================================================
# 1. COMPLETE BASE CNN PIPELINE (MNIST)
# =================================================================
def run_complete_cnn_pipeline():
    """
    Complete CNN pipeline with:
    - Data loading, normalization, splitting
    - Model building (Functional API)
    - Training with history
    - Plotting accuracy/loss
    - Confusion matrix
    """
    print("\n" + "="*60)
    print("SECTION 1: COMPLETE BASE CNN PIPELINE")
    print("="*60)
    
    # --- 1. Load Data ---
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # --- 2. Normalize ---
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # --- 3. Reshape for CNN ---
    x_train_full = x_train_full.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # --- 4. Train-Val Split ---
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, 
        test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    
    # --- 5. Build Model (Functional API) ---
    def build_base_cnn(input_shape=(28, 28, 1), num_classes=10):
        inputs = Input(shape=input_shape)
        
        # Block 1
        x = Conv2D(32, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
        
        # Block 2
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
        
        # Classifier
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        return Model(inputs, outputs, name='Base_CNN')
    
    model = build_base_cnn()
    model.summary()
    
    # --- 6. Compile ---
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # --- 7. Train ---
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(x_val, y_val),
        verbose=1
    )
    
    # --- 8. Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_plots.png')
    plt.show()
    
    # --- 9. Evaluate ---
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
    
    # --- 10. Confusion Matrix ---
    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return model


# =================================================================
# 2. ALEXNET STRUCTURE
# =================================================================
def build_alexnet(input_shape=(224, 224, 3), num_classes=10):
    """
    AlexNet Architecture (2012)
    Key: Large kernels (11x11, 5x5), ReLU, Dropout
    """
    model = Sequential([
        # Block 1: 11x11 conv with stride 4
        layers.Conv2D(96, (11, 11), strides=(4, 4), padding='valid', 
                     activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        
        # Block 2: 5x5 conv
        layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        
        # Blocks 3-5: 3x3 convs
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='AlexNet')
    
    return model


# =================================================================
# 3. VGG16 STRUCTURE
# =================================================================
def build_vgg16(input_shape=(224, 224, 3), num_classes=10):
    """
    VGG16 Architecture (2014)
    Key: All 3x3 convs, 2-3 convs per block, 16 weight layers
    """
    model = Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1: 2x Conv(64)
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2: 2x Conv(128)
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3: 3x Conv(256)
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4: 3x Conv(512)
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 5: 3x Conv(512)
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='VGG16')
    
    return model


# =================================================================
# 4. VGG19 STRUCTURE
# =================================================================
def build_vgg19(input_shape=(224, 224, 3), num_classes=10):
    """
    VGG19 Architecture (2014)
    Key: Same as VGG16 but blocks 3,4,5 have 4 convs instead of 3
    """
    model = Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1: 2x Conv(64)
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2: 2x Conv(128)
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3: 4x Conv(256) - EXTRA LAYER vs VGG16
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),  # Extra!
        layers.MaxPooling2D((2, 2)),
        
        # Block 4: 4x Conv(512) - EXTRA LAYER vs VGG16
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),  # Extra!
        layers.MaxPooling2D((2, 2)),
        
        # Block 5: 4x Conv(512) - EXTRA LAYER vs VGG16
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),  # Extra!
        layers.MaxPooling2D((2, 2)),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='VGG19')
    
    return model


# =================================================================
# 5. INCEPTION BLOCK & MINI GOOGLENET
# =================================================================
def inception_block(x, f1, f3_reduce, f3, f5_reduce, f5, pool_proj):
    """
    Inception Module (GoogLeNet)
    
    Args:
        f1: filters for 1x1 conv path
        f3_reduce: filters for 1x1 before 3x3
        f3: filters for 3x3 conv
        f5_reduce: filters for 1x1 before 5x5
        f5: filters for 5x5 conv
        pool_proj: filters for 1x1 after maxpool
    """
    # Path 1: 1x1 conv
    path1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(x)
    
    # Path 2: 1x1 -> 3x3
    path2 = Conv2D(f3_reduce, (1, 1), padding='same', activation='relu')(x)
    path2 = Conv2D(f3, (3, 3), padding='same', activation='relu')(path2)
    
    # Path 3: 1x1 -> 5x5
    path3 = Conv2D(f5_reduce, (1, 1), padding='same', activation='relu')(x)
    path3 = Conv2D(f5, (5, 5), padding='same', activation='relu')(path3)
    
    # Path 4: MaxPool -> 1x1
    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = Conv2D(pool_proj, (1, 1), padding='same', activation='relu')(path4)
    
    # Concatenate all paths along channel axis
    return Concatenate(axis=-1)([path1, path2, path3, path4])


def build_mini_googlenet(input_shape=(224, 224, 3), num_classes=10):
    """
    Simplified GoogLeNet/Inception v1
    """
    inputs = Input(shape=input_shape)
    
    # Stem
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Inception blocks
    x = inception_block(x, 64, 96, 128, 16, 32, 32)   # 3a
    x = inception_block(x, 128, 128, 192, 32, 96, 64) # 3b
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = inception_block(x, 192, 96, 208, 16, 48, 64)  # 4a
    
    # Classifier (Global Average Pooling instead of Flatten)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name='Mini_GoogLeNet')


# =================================================================
# 6. RESNET BLOCKS
# =================================================================
def basic_residual_block(x, filters, stride=1):
    """
    Basic Residual Block (ResNet-18/34)
    2 convolutions with skip connection
    """
    shortcut = x
    
    # First conv
    x = Conv2D(filters, (3, 3), strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second conv
    x = Conv2D(filters, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Shortcut: projection if dimensions change
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add skip connection
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x


def bottleneck_block(x, filters, stride=1, expansion=4):
    """
    Bottleneck Block (ResNet-50/101/152)
    3 convolutions: 1x1 reduce -> 3x3 -> 1x1 expand
    """
    shortcut = x
    out_filters = filters * expansion
    
    # 1x1: reduce channels
    x = Conv2D(filters, (1, 1), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 3x3: main conv
    x = Conv2D(filters, (3, 3), strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 1x1: expand channels
    x = Conv2D(out_filters, (1, 1), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Shortcut
    if stride != 1 or shortcut.shape[-1] != out_filters:
        shortcut = Conv2D(out_filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x


def build_mini_resnet(input_shape=(224, 224, 3), num_classes=10):
    """
    Minimal ResNet using basic blocks
    """
    inputs = Input(shape=input_shape)
    
    # Stem
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Residual stages
    x = basic_residual_block(x, 64)
    x = basic_residual_block(x, 64)
    
    x = basic_residual_block(x, 128, stride=2)  # Downsample
    x = basic_residual_block(x, 128)
    
    x = basic_residual_block(x, 256, stride=2)  # Downsample
    x = basic_residual_block(x, 256)
    
    # Classifier
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name='Mini_ResNet')


# =================================================================
# 7. PRETRAINED MODEL - COMPLETE PIPELINE (START TO END)
# =================================================================
"""
PRETRAINED MODEL EXPLANATION:
============================
1. Pretrained weights = Model trained on ImageNet (1.2M images, 1000 classes)
2. The model has already learned useful features (edges, textures, shapes)
3. We reuse these features for our own task (Transfer Learning)

IMPORTANT PREPROCESSING:
- ResNet/VGG: Subtract ImageNet mean, channels may be BGR
- MobileNet/EfficientNet: Scale to [-1, 1]
- ALWAYS use the model's preprocess_input() function!

WHEN TO FREEZE vs FINE-TUNE:
- Small dataset → Freeze all (only train new classifier)
- Medium dataset → Freeze early layers, fine-tune later layers
- Large dataset → Fine-tune everything with LOW learning rate
"""

def run_pretrained_pipeline_complete(data_dir):
    """
    COMPLETE PRETRAINED RESNET50 PIPELINE
    =====================================
    This is a full working example from data loading to evaluation.
    
    Args:
        data_dir: Path to dataset folder with class subfolders
                  Example: '/kaggle/input/flower-dataset/dataset'
    """
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    
    print("="*60)
    print("PRETRAINED RESNET50 - COMPLETE PIPELINE")
    print("="*60)
    
    # ============================================================
    # STEP 1: CONFIGURATION
    # ============================================================
    IMG_SIZE = (224, 224)  # ResNet expects 224x224
    BATCH_SIZE = 32
    EPOCHS = 30
    SEED = 42
    
    # ============================================================
    # STEP 2: LOAD DATA FROM DIRECTORY
    # ============================================================
    # Labels are automatically assigned from folder names!
    # Folder structure: dataset/class_a/img1.jpg, dataset/class_b/img2.jpg
    
    print("\n📂 Loading data from directory...")
    
    # 70% training
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.30,
        subset='training',
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # 30% for val+test
    valtest_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.30,
        subset='validation',
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Split 30% into 20% val + 10% test
    valtest_batches = tf.data.experimental.cardinality(valtest_ds).numpy()
    test_batches = valtest_batches // 3
    test_ds = valtest_ds.take(test_batches)
    val_ds = valtest_ds.skip(test_batches)
    
    # Get class info
    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    print(f"✅ Classes: {class_names}")
    print(f"✅ Number of classes: {num_classes}")
    print(f"✅ Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"✅ Val batches: {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"✅ Test batches: {tf.data.experimental.cardinality(test_ds).numpy()}")
    
    # ============================================================
    # STEP 3: DATA AUGMENTATION + PREPROCESSING
    # ============================================================
    print("\n🔄 Setting up augmentation and preprocessing...")
    
    # Augmentation (training only)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
    ])
    
    # IMPORTANT: Use ResNet's preprocessing function!
    # This subtracts ImageNet mean values
    def apply_resnet_preprocessing(images, labels):
        return preprocess_input(images), labels
    
    # Apply augmentation to training
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    
    # Apply ResNet preprocessing to ALL sets
    train_ds = train_ds.map(apply_resnet_preprocessing)
    val_ds = val_ds.map(apply_resnet_preprocessing)
    test_ds = test_ds.map(apply_resnet_preprocessing)
    
    # Optimize pipeline
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    # ============================================================
    # STEP 4: BUILD MODEL WITH PRETRAINED BASE
    # ============================================================
    print("\n🏗️ Building model with pretrained ResNet50...")
    
    # Load pretrained ResNet50 (WITHOUT top classifier)
    base_model = ResNet50(
        weights='imagenet',      # Use ImageNet pretrained weights
        include_top=False,       # Remove the 1000-class classifier
        input_shape=(*IMG_SIZE, 3)
    )
    
    # FREEZE the base model (don't train these layers initially)
    base_model.trainable = False
    
    print(f"✅ Base model layers: {len(base_model.layers)}")
    print(f"✅ Base model trainable: {base_model.trainable}")
    
    # Build full model
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    
    # Pass through frozen base (training=False for BatchNorm)
    x = base_model(inputs, training=False)
    
    # Add custom classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name='Pretrained_ResNet50')
    
    # Print summary
    model.summary()
    
    # ============================================================
    # STEP 5: COMPILE MODEL
    # ============================================================
    print("\n⚙️ Compiling model...")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',  # Integer labels
        metrics=['accuracy']
    )
    
    # ============================================================
    # STEP 6: CALLBACKS
    # ============================================================
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath='best_pretrained_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # ============================================================
    # STEP 7: TRAIN (Feature Extraction Phase)
    # ============================================================
    print("\n🚀 Training (Feature Extraction - Base Frozen)...")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # ============================================================
    # STEP 8: PLOT TRAINING CURVES
    # ============================================================
    print("\n📊 Plotting training curves...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Acc')
    ax1.plot(history.history['val_accuracy'], label='Val Acc')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('pretrained_training_curves.png')
    plt.show()
    
    # ============================================================
    # STEP 9: EVALUATE ON TEST SET
    # ============================================================
    print("\n📝 Evaluating on test set...")
    
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    
    print("\n" + "="*40)
    print(f"✅ TEST ACCURACY: {test_acc*100:.2f}%")
    print(f"✅ TEST LOSS: {test_loss:.4f}")
    print("="*40)
    
    # ============================================================
    # STEP 10: CONFUSION MATRIX
    # ============================================================
    print("\n🔢 Generating confusion matrix...")
    
    # Get predictions
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Pretrained ResNet50')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('pretrained_confusion_matrix.png')
    plt.show()
    
    # ============================================================
    # STEP 11: VISUALIZE PREDICTIONS
    # ============================================================
    print("\n🖼️ Visualizing sample predictions...")
    
    plt.figure(figsize=(15, 10))
    
    for images, labels in test_ds.take(1):
        predictions = model.predict(images, verbose=0)
        
        for i in range(min(9, len(images))):
            plt.subplot(3, 3, i + 1)
            
            # Reverse preprocessing for display (approximate)
            img = images[i].numpy()
            # ResNet preprocessing subtracts mean, we add it back approximately
            img = img + [103.939, 116.779, 123.68]  # BGR means
            img = img[..., ::-1]  # BGR to RGB
            img = np.clip(img / 255.0, 0, 1)
            
            plt.imshow(img)
            
            true_label = class_names[labels[i].numpy()]
            pred_label = class_names[np.argmax(predictions[i])]
            confidence = np.max(predictions[i]) * 100
            
            color = 'green' if true_label == pred_label else 'red'
            plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)", 
                     color=color, fontsize=10)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('pretrained_predictions.png')
    plt.show()
    
    print("\n✅ Pipeline complete!")
    print(f"✅ Model saved as 'best_pretrained_model.keras'")
    
    return model, history, class_names


# --- Simple pretrained model builder (for reference) ---
def build_pretrained_resnet50(input_shape=(224, 224, 3), num_classes=10, trainable=False):
    """
    Transfer Learning with Pretrained ResNet50
    
    Args:
        trainable: If False, freeze base model (feature extraction)
                   If True, allow fine-tuning
    """
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    
    # Load pretrained base (without top classifier)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze or unfreeze base
    base_model.trainable = trainable
    
    # Build model
    inputs = Input(shape=input_shape)
    # Note: training=False keeps BatchNorm in inference mode
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='Pretrained_ResNet50')
    
    return model, preprocess_input  # Return preprocessing function too!


# =================================================================
# 8. IMAGE LOADING METHODS - DETAILED EXPLANATION
# =================================================================
"""
HOW LABELS WORK FROM DIRECTORY:
===============================
When you use image_dataset_from_directory(), labels are AUTOMATICALLY 
assigned based on FOLDER NAMES (alphabetically sorted).

Example folder structure:
    dataset/
        cat/        ← Label 0 (alphabetically first)
            cat1.jpg
            cat2.jpg
        dog/        ← Label 1 
            dog1.jpg
            dog2.jpg
        elephant/   ← Label 2
            ele1.jpg

The function:
1. Scans subfolders → class_names = ['cat', 'dog', 'elephant']
2. Assigns integer labels: cat=0, dog=1, elephant=2
3. Returns (image, label) pairs as tf.data.Dataset

You can access class names via: dataset.class_names
"""

# --- Method 1: From Keras datasets ---
def load_keras_dataset(dataset_name='mnist'):
    """Load built-in datasets"""
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dim for grayscale
    if len(x_train.shape) == 3:
        x_train = x_train.reshape(*x_train.shape, 1)
        x_test = x_test.reshape(*x_test.shape, 1)
    
    return (x_train, y_train), (x_test, y_test)


# --- Method 2: From folder structure (DETAILED) ---
def load_from_directory_complete(data_dir, img_size=(224, 224), batch_size=32):
    """
    COMPLETE EXAMPLE: Load from folder with train/val/test split
    
    Folder structure:
        dataset/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img3.jpg
                img4.jpg
    
    Labels are AUTOMATICALLY assigned by folder name (alphabetically):
        class_a → 0
        class_b → 1
    """
    
    # ============ STEP 1: Load with 70-30 split ============
    # Get 70% for training
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.30,    # Hold out 30%
        subset='training',        # This is the 70%
        seed=42,                  # Same seed = same split
        image_size=img_size,      # Resize all images
        batch_size=batch_size,
        label_mode='int'          # Integer labels (0, 1, 2...)
    )
    
    # Get 30% for validation+test
    valtest_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.30,
        subset='validation',      # This is the 30%
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int'
    )
    
    # ============ STEP 2: Split val/test (30% → 20% val + 10% test) ============
    valtest_batches = tf.data.experimental.cardinality(valtest_ds).numpy()
    test_batches = valtest_batches // 3   # ~10% of total
    
    test_ds = valtest_ds.take(test_batches)    # First portion = test
    val_ds = valtest_ds.skip(test_batches)     # Rest = validation
    
    # ============ STEP 3: Get class names (IMPORTANT!) ============
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Classes found: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    # ============ STEP 4: Preprocessing ============
    # Normalization layer
    rescale = Rescaling(1./255)
    
    # Data augmentation (training only!)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])
    
    # Apply augmentation + normalization to training
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    train_ds = train_ds.map(lambda x, y: (rescale(x), y))
    
    # Only normalization for val/test (NO augmentation!)
    val_ds = val_ds.map(lambda x, y: (rescale(x), y))
    test_ds = test_ds.map(lambda x, y: (rescale(x), y))
    
    # ============ STEP 5: Optimize pipeline ============
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    print(f"Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"Val batches: {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"Test batches: {tf.data.experimental.cardinality(test_ds).numpy()}")
    
    return train_ds, val_ds, test_ds, class_names


# =================================================================
# HOW TO TEST WITH DIRECTORY-LOADED DATA
# =================================================================
"""
TESTING EXPLAINED:
=================
When you load from directory, the dataset is a tf.data.Dataset of (images, labels).
Testing is done using model.evaluate() which automatically:
1. Iterates through all batches in test_ds
2. Computes loss and metrics
3. Returns [loss, accuracy]

For predictions:
- model.predict(test_ds) → Returns probabilities
- np.argmax(predictions, axis=1) → Converts to class indices
"""

def test_model_on_directory_data(model, test_ds, class_names):
    """
    Complete testing pipeline for directory-loaded data
    
    Args:
        model: Trained Keras model
        test_ds: tf.data.Dataset with (images, labels)
        class_names: List of class names ['cat', 'dog', ...]
    """
    print("\n" + "="*50)
    print("TESTING MODEL")
    print("="*50)
    
    # ============ 1. EVALUATE (Get loss + accuracy) ============
    print("\n📊 Running model.evaluate()...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    
    print(f"\n✅ Test Loss: {test_loss:.4f}")
    print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
    
    # ============ 2. GET PREDICTIONS ============
    print("\n🔮 Getting predictions...")
    
    y_true = []  # Actual labels
    y_pred = []  # Predicted labels
    
    # Loop through test dataset
    for images, labels in test_ds:
        # Get predictions (probabilities)
        predictions = model.predict(images, verbose=0)
        
        # Convert probabilities to class indices
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Store results
        y_pred.extend(predicted_classes)
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # ============ 3. CONFUSION MATRIX ============
    print("\n📊 Generating confusion matrix...")
    from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Print classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # ============ 4. PER-CLASS ACCURACY ============
    print("\n📊 Per-class accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred[class_mask] == i) * 100
            print(f"  {class_name}: {class_acc:.1f}%")
    
    return y_true, y_pred


def visualize_dataset_samples(dataset, class_names, n_samples=9):
    """Visualize samples from dataset with their labels"""
    plt.figure(figsize=(10, 10))
    
    # Take one batch
    for images, labels in dataset.take(1):
        for i in range(min(n_samples, len(images))):
            plt.subplot(3, 3, i + 1)
            # Images are normalized [0,1], matplotlib handles this
            plt.imshow(images[i].numpy())
            # Convert integer label to class name
            label_idx = labels[i].numpy()
            plt.title(f"{class_names[label_idx]} (label={label_idx})")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# --- Method 3: From CSV with paths ---
def load_from_csv(csv_path, img_size=(224, 224), batch_size=32):
    """
    Load from CSV:
    image_path,label
    images/cat1.jpg,0
    images/dog1.jpg,1
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = img / 255.0
        return img, label
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (df['image_path'].values, df['label'].values)
    )
    dataset = dataset.map(load_and_preprocess)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


# --- Data Augmentation ---
def create_augmentation_layer():
    """Standard augmentation for training"""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ])


# =================================================================
# 9. U-NET FOR SEGMENTATION
# =================================================================
def build_unet(input_shape=(256, 256, 3), num_classes=34):
    """
    U-Net Architecture for Semantic Segmentation
    
    Structure:
    - Encoder: 4 blocks (down-sampling)
    - Bottleneck
    - Decoder: 4 blocks (up-sampling with skip connections)
    """
    inputs = Input(shape=input_shape)
    
    # ============ ENCODER ============
    # Block 1
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    # Block 2
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Block 3
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Block 4
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # ============ BOTTLENECK ============
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # ============ DECODER ============
    # Block 6: Upsample + Skip from c4
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])  # SKIP CONNECTION
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    # Block 7: Upsample + Skip from c3
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    # Block 8: Upsample + Skip from c2
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    # Block 9: Upsample + Skip from c1
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    # ============ OUTPUT ============
    # 1x1 conv for per-pixel classification
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    
    return Model(inputs, outputs, name='U-Net')


def load_segmentation_data(img_dir, mask_dir, img_height=256, img_width=256, batch_size=8):
    """
    Load image-mask pairs for segmentation
    """
    import glob
    import os
    
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    
    def load_data(img_path, mask_path):
        # Load image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (img_height, img_width))
        img = tf.cast(img, tf.float32) / 255.0
        
        # Load mask (NEAREST interpolation to preserve labels!)
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, (img_height, img_width), method='nearest')
        mask = tf.cast(mask, tf.int32)
        
        return img, mask
    
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


# =================================================================
# COMPLETE EXAMPLE: TRAINING FROM DIRECTORY (EXAM-READY)
# =================================================================
def complete_training_from_directory_example():
    """
    COMPLETE WORKING EXAMPLE - Copy this for exam!
    Shows: Load from folder → Train → Test → Confusion Matrix
    
    Change 'data_dir' to your dataset path.
    """
    print("="*60)
    print("COMPLETE EXAMPLE: DIRECTORY → TRAIN → TEST")
    print("="*60)
    
    # ============ CONFIG ============
    data_dir = '/kaggle/input/flower-dataset/dataset'  # CHANGE THIS!
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 10
    
    # ============ LOAD DATA ============
    # Labels come from FOLDER NAMES automatically!
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.30,
        subset='training',
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    valtest_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.30,
        subset='validation',
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Split val/test
    valtest_batches = tf.data.experimental.cardinality(valtest_ds).numpy()
    test_ds = valtest_ds.take(valtest_batches // 3)
    val_ds = valtest_ds.skip(valtest_batches // 3)
    
    # Get class info
    class_names = train_ds.class_names  # ['daisy', 'rose', 'sunflower', ...]
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    
    # ============ NORMALIZE ============
    rescale = Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (rescale(x), y))
    val_ds = val_ds.map(lambda x, y: (rescale(x), y))
    test_ds = test_ds.map(lambda x, y: (rescale(x), y))
    
    # ============ BUILD MODEL ============
    inputs = Input(shape=(*IMG_SIZE, 3))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # ============ TRAIN ============
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    
    # ============ TEST ============
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
    
    # ============ PREDICTIONS ============
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
    
    # ============ CONFUSION MATRIX ============
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()
    
    return model


# =================================================================
# MAIN: Show model summaries
# =================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("CNN ARCHITECTURES SUMMARY")
    print("="*60)
    
    # ============ UNCOMMENT TO RUN EXAMPLES ============
    # 1. Basic CNN on MNIST
    # run_complete_cnn_pipeline()
    
    # 2. Pretrained ResNet50 (needs data_dir)
    # run_pretrained_pipeline_complete('/path/to/dataset')
    
    # 3. Simple training from directory
    # complete_training_from_directory_example()
    
    # ============ SHOW ARCHITECTURE SUMMARIES ============
    print("\n--- AlexNet ---")
    alexnet = build_alexnet(num_classes=10)
    print(f"Total params: {alexnet.count_params():,}")
    
    print("\n--- VGG16 ---")
    vgg16 = build_vgg16(num_classes=10)
    print(f"Total params: {vgg16.count_params():,}")
    
    print("\n--- VGG19 ---")
    vgg19 = build_vgg19(num_classes=10)
    print(f"Total params: {vgg19.count_params():,}")
    
    print("\n--- Mini GoogLeNet ---")
    googlenet = build_mini_googlenet(num_classes=10)
    print(f"Total params: {googlenet.count_params():,}")
    
    print("\n--- Mini ResNet ---")
    resnet = build_mini_resnet(num_classes=10)
    print(f"Total params: {resnet.count_params():,}")
    
    print("\n--- U-Net ---")
    unet = build_unet(num_classes=34)
    print(f"Total params: {unet.count_params():,}")
    
    print("\n" + "="*60)
    print("✅ All architectures defined successfully!")
    print("="*60)
    
    # Print quick reference
    print("""
    QUICK REFERENCE:
    ================
    
    LOADING FROM DIRECTORY:
    -----------------------
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'dataset/',           # Folder with class subfolders
        validation_split=0.2,
        subset='training',
        seed=42,
        image_size=(224, 224),
        batch_size=32
    )
    class_names = train_ds.class_names  # ['cat', 'dog', ...]
    
    LABELS ARE AUTOMATIC!
    - Folder names = class names
    - Alphabetically sorted = label indices
    - 'cat' folder → label 0, 'dog' folder → label 1
    
    TESTING:
    --------
    # Simple evaluation
    loss, acc = model.evaluate(test_ds)
    
    # Get predictions
    for images, labels in test_ds:
        preds = model.predict(images)
        pred_classes = np.argmax(preds, axis=1)
    
    ARCHITECTURES:
    ==============
    AlexNet:   11x11, 5x5, 3x3 kernels, ReLU, first GPU CNN
    VGG16:     All 3x3 kernels, 16 layers deep
    VGG19:     VGG16 + 3 extra conv layers in blocks 3,4,5
    GoogLeNet: Inception blocks (parallel 1x1, 3x3, 5x5)
    ResNet:    Skip connections (x + F(x))
    U-Net:     Encoder-Decoder with skip connections for segmentation
    
    KEY FORMULAS:
    =============
    Output size = (W - K + 2P) / S + 1
    
    Conv2D params = (K × K × C_in + 1) × C_out
    Dense params = (inputs + 1) × outputs
    """)
