from tgs.models.UNet import get_unet
from tgs.data.images import HEIGHT, WIDTH, Image, ImageSet
from tgs.data.split import get_train_val_ids
from tgs.utils.mlflow import find_or_create_experiment
from tgs.utils.metrics import tf_mean_iou

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, History

import mlflow
import os
import json

# TODO MLFlow
# TODO Parse arguments


if __name__ == '__main__':
    EXP_NAME = "UNet"
    exp = find_or_create_experiment(EXP_NAME, mlflow.tracking.list_experiments())
    if isinstance(exp, list):
        raise TypeError("Multiple experiment with that name where found.")  # Not sure it's possible tho

    with mlflow.start_run(experiment_id=exp.experiment_id):
        active_run = mlflow.active_run()

        # Idea Could be cool to have a wrapper for this...
        # Parameters

        loss = "binary_crossentropy"
        mlflow.log_param("loss", loss)

        optimizer = "SGD"
        mlflow.log_param("optimizer", optimizer)

        es_patience = 3
        mlflow.log_param("es_parience", es_patience)

        exp_type = "unittest"
        mlflow.log_param("exp_type", exp_type)

        batch_size = 32
        mlflow.log_param("batch_size", batch_size)

        epochs = 1000
        mlflow.log_param("epochs", epochs)

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=es_patience,
                                       verbose=1, mode='auto')
        tensorboard = TensorBoard(log_dir=active_run.info.artifact_uri, histogram_freq=0,
                                  write_graph=True, write_images=True)
        checkpoint = ModelCheckpoint(active_run.info.artifact_uri + "/model.h5",
                                     monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)
        history = History()
        callbacks = [early_stopping, tensorboard, checkpoint, history]

        model = get_unet((HEIGHT, WIDTH, 1))

        imgs_train = list()
        imgs_val = list()

        img_path = "data/raw/train"

        # TODO Later - Cache ImageSet object numpy save
        train_ids, val_ids = get_train_val_ids(img_path + "/images/")
        for img_id in os.listdir(img_path + "/images/"):
            if img_id in train_ids:
                imgs_train.append(Image(img_id, img_path))
            else:
                imgs_val.append(Image(img_id, img_path))

        trainset = ImageSet(imgs_train, HEIGHT, WIDTH, 1)
        valset = ImageSet(imgs_val, HEIGHT, WIDTH, 1)

        X_train, y_train = trainset.get_x_y()
        X_val, y_val = valset.get_x_y()

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', tf_mean_iou])
        print(model.summary())

        with open(active_run.info.artifact_uri + "/network_architecture.json", "w") as f:
            json.dump(model.to_json(), f)

        # TODO unit testesting should be done differently
        if exp_type == "unittest":
            n = 16
        else:
            n = len(X_train)

        # Launch training

        model.fit(x=X_train[:n], y=y_train[:n], epochs=epochs, verbose=1, callbacks=callbacks,
                  validation_data=(X_val, y_val), batch_size=batch_size)

        for metric in history.history:
            for i in range(len(history.history[metric])):
                mlflow.log_metric(metric, history.history[metric][i])
        mlflow.log_metric("trained_epoch", len(history.history["loss"]))

