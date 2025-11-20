# # fl_client.py
# import flwr as fl
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras import layers, models

# # -------- CHANGE THIS FOR EACH CLIENT ----------
# data_dir = "data/disease/rice/farm_1"
# # Example for client 2:
# # data_dir = "data/disease/rice/farm_2"
# # ------------------------------------------------


# def load_data():
#     datagen = ImageDataGenerator(
#         rescale=1.0 / 255,
#         validation_split=0.2,
#         rotation_range=20,
#         zoom_range=0.2,
#         horizontal_flip=True
#     )

#     train = datagen.flow_from_directory(
#         data_dir,
#         target_size=(128, 128),
#         batch_size=16,
#         class_mode='categorical',
#         subset='training',
#         shuffle=True
#     )

#     val = datagen.flow_from_directory(
#         data_dir,
#         target_size=(128, 128),
#         batch_size=8,
#         class_mode='categorical',
#         subset='validation'
#     )

#     return train, val


# # ---------- SAME MODEL AS SERVER ----------
# def build_model(num_classes):
#     model = models.Sequential([
#         layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
#         layers.MaxPooling2D(2, 2),

#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.MaxPooling2D(2, 2),

#         layers.Conv2D(128, (3, 3), activation='relu'),
#         layers.MaxPooling2D(2, 2),

#         layers.Flatten(),
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.4),
#         layers.Dense(num_classes, activation='softmax'),
#     ])

#     model.compile(
#         optimizer='adam',
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model


# # ---------- FLOWER CLIENT ----------
# class FlowerClient(fl.client.NumPyClient):
#     def __init__(self, model, train, val):
#         self.model = model
#         self.train = train
#         self.val = val

#     def get_parameters(self, config):
#         # Send local model weights to server
#         return self.model.get_weights()

#     def fit(self, parameters, config):
#         # Receive global weights from server
#         self.model.set_weights(parameters)

#         # Train locally
#         self.model.fit(self.train, epochs=3, verbose=1)

#         # Return updated weights
#         return self.model.get_weights(), len(self.train), {}

#     def evaluate(self, parameters, config):
#         self.model.set_weights(parameters)
#         loss, acc = self.model.evaluate(self.val, verbose=0)
#         return loss, len(self.val), {"accuracy": acc}


# if __name__ == "__main__":
#     train, val = load_data()
#     model = build_model(num_classes=len(train.class_indices))
#     fl.client.start_numpy_client(
#         server_address="127.0.0.1:8080",
#         client=FlowerClient(model, train, val)
#     )




# fl_client_1.py
import flwr as fl
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

data_dir = "data/disease/rice/farm_1"  # client 1 dataset path

def load_data():
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1.0/255,
                                 rotation_range=20, zoom_range=0.2, horizontal_flip=True)
    train = datagen.flow_from_directory(
        data_dir, target_size=(128,128), batch_size=8,
        class_mode="categorical", subset="training", shuffle=True
    )
    val = datagen.flow_from_directory(
        data_dir, target_size=(128,128), batch_size=8,
        class_mode="categorical", subset="validation", shuffle=False
    )
    return train, val

def build_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train, val):
        self.model = model
        self.train = train
        self.val = val

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.train, epochs=5, verbose=1)   # increase epochs if needed
        return self.model.get_weights(), len(self.train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.val, verbose=0)
        return loss, len(self.val), {"accuracy": acc}

if __name__ == "__main__":
    train, val = load_data()
    model = build_model(len(train.class_indices))
    print("[Client 1] classes:", train.class_indices)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient(model, train, val))
