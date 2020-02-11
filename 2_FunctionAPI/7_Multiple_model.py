from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

def autoencoder():
    # share layer mdoel
    encoder_input = keras.Input(shape = (28,28,1),name = 'img')
    x = layers.Conv2D(16,3,activation = 'relu')(encoder_input)
    x = layers.Conv2D(32,3,activation = 'relu')(x)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Conv2D(32,3,activation = 'relu')(x)
    x = layers.Conv2D(16,3,activation = 'relu')(x)
    encoder_output = layers.GlobalMaxPooling2D()(x)

    encoder = keras.Model(encoder_input,encoder_output,name='encoder')
    encoder.summary()

    x = layers.Reshape((4,4,1))(encoder_output)
    x = layers.Conv2DTranspose(16,3,activation='relu')(x)
    x = layers.Conv2DTranspose(32,3,activation='relu')(x)
    x = layers.UpSampling2D(3)(x)
    x = layers.Conv2DTranspose(16,3,activation='relu')(x)
    decoder_output = layers.Conv2DTranspose(1,3,activation='relu')(x)

    autoencoder = keras.Model(encoder_input,decoder_output,name='autoencoder')
    autoencoder.summary()

    # return autoencoder

def multi_model():
    # multi input - multi models - chain them together
    encoder_input = keras.Input(shape = (28,28,1),name = 'original_img')
    x = layers.Conv2D(16,3,activation = 'relu')(encoder_input)
    x = layers.Conv2D(32,3,activation = 'relu')(x)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Conv2D(32,3,activation = 'relu')(x)
    x = layers.Conv2D(16,3,activation = 'relu')(x)
    encoder_output = layers.GlobalMaxPooling2D()(x)

    encoder = keras.Model(encoder_input,encoder_output,name='encoder')
    encoder.summary()

    decoder_input = keras.Input(shape = (16,), name = 'encoded_img')
    x = layers.Reshape((4,4,1))(decoder_input)
    x = layers.Conv2DTranspose(16,3,activation='relu')(x)
    x = layers.Conv2DTranspose(32,3,activation='relu')(x)
    x = layers.UpSampling2D(3)(x)
    x = layers.Conv2DTranspose(16,3,activation='relu')(x)
    decoder_output = layers.Conv2DTranspose(1,3,activation='relu')(x)

    decoder = keras.Model(decoder_input,decoder_output,name='decoder')
    decoder.summary()

    autoencoder_input = keras.Input(shape = (28,28,1),name = 'img')
    encode_img = encoder(autoencoder_input)
    decoder_img = decoder(encode_img)
    autoencoder = keras.Model(autoencoder_input,decoder_img,name='autoencoder')
    autoencoder.summary()

def multi_input_output():
    num_tags = 12
    num_words = 10000
    num_departments = 4

    title_input = keras.Input(shape = (None,) , name = 'title')
    body_input = keras.Input(shape = (None,) , name = 'body')
    tags_input = keras.Input(shape = (num_tags,) , name = 'tags')

    title_features = layers.Embedding(num_words,64)(title_input)
    body_features = layers.Embedding(num_words, 64)(body_input)

    title_features = layers.LSTM(128)(title_features)
    body_features = layers.LSTM(32)(body_features)

    x = layers.concatenate([title_features, body_features, tags_input])

    priority_pred = layers.Dense(1, activation='sigmoid', name='priority')(x)
    department_pred = layers.Dense(num_departments, activation='softmax', name='department')(x)

    model = keras.Model(inputs=[title_input, body_input, tags_input],
                    outputs=[priority_pred, department_pred])
    model.summary()

    keras.utils.plot_model(model,"multiIO.png",show_shapes = True)

    model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
              loss=['binary_crossentropy', 'categorical_crossentropy'],
              loss_weights=[1., 0.2])
    
    callbacks = [keras.callbacks.TensorBoard(log_dir = './logs')]

    # Dummy input data
    title_data = np.random.randint(num_words, size=(1280, 10))
    body_data = np.random.randint(num_words, size=(1280, 100))
    tags_data = np.random.randint(2, size=(1280, num_tags)).astype('float32')
    # Dummy target data
    priority_targets = np.random.random(size=(1280, 1))
    dept_targets = np.random.randint(2, size=(1280, num_departments))

    model.fit({'title': title_data, 'body': body_data, 'tags': tags_data},
            {'priority': priority_targets, 'department': dept_targets},
            epochs=2,
            batch_size=32,callbacks=callbacks)


if __name__ == "__main__":
    multi_input_output()