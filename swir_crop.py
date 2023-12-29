##### 임포트
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow_addons.metrics
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score

##### GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 기본 설정
args = {
    "data_folder": "D:/waper/crop_data/",
    "graphs_folder": "D:/waper/crop/graph/",
    "model_save_folder": "D:/waper/crop/trained_model/",
    "epoch": 50,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "img_size": (300, 300),
    "SEED": 41
}

# Seed 고정
tf.random.set_seed(args["SEED"])

# 데이터셋 로드
datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.3,
    fill_mode='nearest'
)


train_generator = datagen.flow_from_directory(
    args["data_folder"],
    target_size=args["img_size"],
    batch_size=args["batch_size"],
    class_mode='binary',
    subset='training'
)

test_generator = datagen.flow_from_directory(
    args["data_folder"],
    target_size=args["img_size"],
    batch_size=args["batch_size"],
    class_mode='binary',
    subset='validation'
)

##### 모델 로드
base_model = ResNet50(weights='imagenet', include_top=True, input_tensor=Input(shape=(300, 300, 3)))      # 원본 이미지

# 마지막 계층 제거
x = base_model.layers[-2].output

# 이진 분류를 위한 새로운 계층 추가
predictions = Dense(1, activation='sigmoid')(x)

# 새로운 모델 생성
model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=args["learning_rate"]), loss='binary_crossentropy',
              metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='AUC'), tensorflow_addons.metrics.F1Score(num_classes=1, average='macro')])

# 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
model_checkpoint = ModelCheckpoint(filepath=args["model_save_folder"] + 'crop.h5', save_best_only=True, monitor='val_loss', verbose=1)

# 모델 훈련
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // args["batch_size"],
    epochs=args["epoch"],
    validation_data=test_generator,
    validation_steps=test_generator.samples // args["batch_size"],
    callbacks=[early_stopping, model_checkpoint]
)

# 정확도 그래프
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)
plt.figure()
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(args["graphs_folder"] + 'crop' + f"acc.png")
plt.show()

# 손실 그래프
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure()
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(args["graphs_folder"] + 'crop' + f"loss.png")
plt.show()

# 평가 및 시각화
def evaluate_model(model, test_generator):
    y_trues = test_generator.classes
    y_preds = model.predict(test_generator, steps=len(test_generator))
    y_preds = y_preds.squeeze()

    f1 = f1_score(y_trues, y_preds > 0.5)
    auc = roc_auc_score(y_trues, y_preds)
    accuracy = accuracy_score(y_trues, y_preds > 0.5)

    print(f'F1 score: {f1:.4f}, AUC: {auc:.4f}, Accuracy: {accuracy:.4f}')

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_trues, y_score=y_preds)
    roc_auc = roc_auc_score(y_true=y_trues, y_score=y_preds, average='macro')
    plt.title(f"Receiver Operating Characteristic")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label="AUC = %0.4f" % roc_auc)
    plt.plot([0, 1], [1, 1], 'y--')
    plt.plot([0, 1], [0, 1], 'r--')

    plt.legend(loc="lower right")
    plt.savefig(args["graphs_folder"] + 'crop' + f"auc.png")
    plt.show()

# 모델 평가
evaluate_model(model, test_generator)
