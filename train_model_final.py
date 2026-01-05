import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import pickle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 3

# Paths
TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'
MODEL_PATH = 'models/breast_cancer_model.h5'
WEIGHTS_PATH = 'models/model_weights.h5'

os.makedirs('models', exist_ok=True)

print("="*70)
print(" BREAST CANCER CLASSIFIER - FINAL OPTIMIZED VERSION")
print("="*70)

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\nClasses: {train_generator.class_indices}")
print(f"Training: {train_generator.samples} | Validation: {validation_generator.samples} | Test: {test_generator.samples}\n")

# Class weights
class_weights_array = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights = {
    i: float(min(class_weights_array[i], 1.8)) 
    for i in range(len(class_weights_array))
}

print("Class Weights:")
for name, idx in train_generator.class_indices.items():
    print(f"  {name.capitalize():12}: {class_weights[idx]:.2f}")

# Build model
def build_model():
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        layer.trainable = True
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

print("\nBuilding model...\n")
model = build_model()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    )
]

# Train
print("="*70)
print("TRAINING")
print("="*70)
print(f"Max epochs: {EPOCHS}")
print(f"Estimated time: 15-20 minutes\n")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# Save
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

try:
    model.save(MODEL_PATH, save_format='h5', include_optimizer=False)
    print(f"✅ Model saved: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️  H5 save failed, saving weights...")
    try:
        model.save_weights(WEIGHTS_PATH)
        print(f"✅ Weights saved: {WEIGHTS_PATH}")
        
        model_json = model.to_json()
        with open('models/model_architecture.json', 'w') as f:
            f.write(model_json)
        print(f"✅ Architecture saved")
    except Exception as e2:
        print(f"❌ Save failed: {e2}")

with open('models/class_indices.pkl', 'wb') as f:
    pickle.dump(train_generator.class_indices, f)
print(f"✅ Class indices saved")

# Evaluate
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

test_generator.reset()
y_pred = model.predict(test_generator, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

print(f"\n✅ Test Accuracy: {test_accuracy*100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=list(train_generator.class_indices.keys())))

cm = confusion_matrix(y_true, y_pred_classes)
print("\nConfusion Matrix:")
class_names = list(train_generator.class_indices.keys())
print("       BEN  MAL  NOR")
for i, name in enumerate(class_names):
    print(f"{name[:6]:6}", cm[i])

print("\nPer-Class Accuracy:")
for i, name in enumerate(class_names):
    total = int(cm.sum(axis=1)[i])
    if total > 0:
        correct = cm[i, i]
        acc = correct / total * 100
        print(f"  {name.capitalize():12}: {acc:5.1f}% ({correct}/{total} correct)")

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"📊 Best Val Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
print(f"📊 Test Accuracy: {test_accuracy*100:.2f}%")
print(f"⏱️  Epochs trained: {len(history.history['loss'])}")
print("="*70)

if test_accuracy < 0.45:
    print("\n⚠️  WARNING: Low accuracy!")
    print("Dataset may be too small - consider collecting more data")

print("\n🚀 Run app: streamlit run app.py\n")
