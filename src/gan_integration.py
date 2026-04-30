import os

import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from config import (
    ENCODER_PATH,
    GAN_DISCRIMINATOR,
    GAN_GENERATOR_BEST,
    SCALER_PATH,
)
from src.dataset_builder import build_multi_dataset


def _make_generator(noise_dim, num_classes, feature_dim):
    noise_input = tf.keras.layers.Input(shape=(noise_dim,))
    label_input = tf.keras.layers.Input(shape=(1,), dtype="int32")
    label_embed = tf.keras.layers.Embedding(num_classes, 32)(label_input)
    label_embed = tf.keras.layers.Flatten()(label_embed)
    x = tf.keras.layers.Concatenate()([noise_input, label_embed])
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    output = tf.keras.layers.Dense(feature_dim, activation="tanh")(x)
    return tf.keras.Model([noise_input, label_input], output, name="feature_generator")


def _make_discriminator(num_classes, feature_dim):
    feature_input = tf.keras.layers.Input(shape=(feature_dim,))
    label_input = tf.keras.layers.Input(shape=(1,), dtype="int32")
    label_embed = tf.keras.layers.Embedding(num_classes, 32)(label_input)
    label_embed = tf.keras.layers.Flatten()(label_embed)
    x = tf.keras.layers.Concatenate()([feature_input, label_embed])
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model([feature_input, label_input], output, name="feature_discriminator")


def train_conditional_gan(
    X_train_scaled,
    y_train_encoded,
    *,
    noise_dim=64,
    batch_size=128,
    epochs=300,
    verbose=1,
):
    num_classes = len(np.unique(y_train_encoded))
    feature_dim = X_train_scaled.shape[1]
    X_train_scaled = np.asarray(X_train_scaled, dtype=np.float32)
    y_train_encoded = np.asarray(y_train_encoded, dtype=np.int32)

    generator = _make_generator(noise_dim, num_classes, feature_dim)
    discriminator = _make_discriminator(num_classes, feature_dim)

    bce = tf.keras.losses.BinaryCrossentropy()
    g_opt = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train_encoded))
    dataset = dataset.shuffle(len(X_train_scaled), reshuffle_each_iteration=True).batch(
        batch_size, drop_remainder=False
    )

    history = {"d_loss": [], "g_loss": []}

    for epoch in range(epochs):
        d_losses = []
        g_losses = []
        for real_features, labels in dataset:
            batch_len = tf.shape(real_features)[0]
            noise = tf.random.normal((batch_len, noise_dim))
            fake_features = generator([noise, labels], training=True)

            real_targets = tf.ones((batch_len, 1)) * 0.9
            fake_targets = tf.zeros((batch_len, 1))

            with tf.GradientTape() as d_tape:
                real_pred = discriminator([real_features, labels], training=True)
                fake_pred = discriminator([fake_features, labels], training=True)
                d_loss_real = bce(real_targets, real_pred)
                d_loss_fake = bce(fake_targets, fake_pred)
                d_loss = d_loss_real + d_loss_fake

            d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
            d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

            misleading_targets = tf.ones((batch_len, 1))
            noise = tf.random.normal((batch_len, noise_dim))
            sampled_labels = tf.random.uniform(
                (batch_len, 1), minval=0, maxval=num_classes, dtype=tf.int32
            )

            with tf.GradientTape() as g_tape:
                generated = generator([noise, sampled_labels], training=True)
                pred = discriminator([generated, sampled_labels], training=True)
                g_loss = bce(misleading_targets, pred)

            g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
            g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))

        history["d_loss"].append(float(np.mean(d_losses)))
        history["g_loss"].append(float(np.mean(g_losses)))
        if verbose and ((epoch + 1) % 25 == 0 or epoch == 0 or epoch == epochs - 1):
            print(
                f"[GAN] epoch={epoch + 1:04d}/{epochs} "
                f"d_loss={history['d_loss'][-1]:.4f} g_loss={history['g_loss'][-1]:.4f}",
                flush=True,
            )

    return generator, discriminator, history


def generate_emotion_features(emotion_label, num_samples, *, generator, label_encoder, noise_dim=64):
    emotion_idx = label_encoder.transform([emotion_label])[0]
    noise = tf.random.normal((num_samples, noise_dim))
    labels = tf.fill((num_samples, 1), tf.cast(emotion_idx, tf.int32))
    generated = generator([noise, labels], training=False).numpy()
    return generated.astype(np.float32)


def augment_existing_data(
    X_train_scaled,
    y_train_labels,
    *,
    augmentation_factor=1.0,
    noise_dim=64,
    gan_epochs=300,
    gan_batch_size=128,
    verbose=1,
):
    y_train_labels = np.asarray(y_train_labels)
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train_labels)
    y_train_encoded = label_encoder.transform(y_train_labels)

    gan_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_gan = gan_scaler.fit_transform(X_train_scaled)

    generator, discriminator, history = train_conditional_gan(
        X_gan,
        y_train_encoded,
        noise_dim=noise_dim,
        batch_size=gan_batch_size,
        epochs=gan_epochs,
        verbose=verbose,
    )

    counts = {label: int(np.sum(y_train_labels == label)) for label in label_encoder.classes_}
    target_count = max(counts.values())

    synthetic_features = []
    synthetic_labels = []

    for emotion in label_encoder.classes_:
        current = counts[emotion]
        desired = max(current, int(target_count * augmentation_factor))
        to_generate = max(0, desired - current)
        if to_generate == 0:
            continue
        generated = generate_emotion_features(
            emotion,
            to_generate,
            generator=generator,
            label_encoder=label_encoder,
            noise_dim=noise_dim,
        )
        synthetic_features.append(generated)
        synthetic_labels.extend([emotion] * to_generate)

    if synthetic_features:
        synthetic_features = np.vstack(synthetic_features)
        synthetic_features = gan_scaler.inverse_transform(synthetic_features).astype(np.float32)
        synthetic_labels = np.asarray(synthetic_labels)
    else:
        synthetic_features = np.empty((0, X_train_scaled.shape[1]), dtype=np.float32)
        synthetic_labels = np.empty((0,), dtype=y_train_labels.dtype)

    os.makedirs(os.path.dirname(GAN_GENERATOR_BEST), exist_ok=True)
    generator.save(GAN_GENERATOR_BEST, include_optimizer=False)
    discriminator.save(GAN_DISCRIMINATOR, include_optimizer=False)

    return synthetic_features, synthetic_labels, {
        "generator": generator,
        "discriminator": discriminator,
        "label_encoder": label_encoder,
        "gan_scaler": gan_scaler,
        "history": history,
    }


def prepare_gan_augmented_dataset(
    dataset_paths,
    *,
    use_gan=True,
    augmentation_factor=1.0,
    test_size=0.2,
    random_state=42,
    gan_epochs=300,
    gan_batch_size=128,
    return_artifacts=False,
    verbose=1,
):
    X, y = build_multi_dataset(dataset_paths)
    y = np.asarray(y)

    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    y_encoded = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    artifacts = {"label_encoder": label_encoder, "scaler": scaler}

    if use_gan:
        X_syn, y_syn, gan_artifacts = augment_existing_data(
            X_train_scaled,
            y_train,
            augmentation_factor=augmentation_factor,
            gan_epochs=gan_epochs,
            gan_batch_size=gan_batch_size,
            verbose=verbose,
        )
        if X_syn.shape[0] > 0:
            X_train_scaled = np.vstack([X_train_scaled, X_syn]).astype(np.float32)
            y_train = np.concatenate([y_train, y_syn])
            order = np.random.default_rng(random_state).permutation(len(X_train_scaled))
            X_train_scaled = X_train_scaled[order]
            y_train = y_train[order]
        artifacts.update(gan_artifacts)

    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    if return_artifacts:
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, artifacts
    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded


def compare_with_without_gan(dataset_paths, *, augmentation_factor=1.0, gan_epochs=300):
    base = prepare_gan_augmented_dataset(
        dataset_paths,
        use_gan=False,
        augmentation_factor=augmentation_factor,
        gan_epochs=gan_epochs,
        return_artifacts=True,
        verbose=0,
    )
    gan = prepare_gan_augmented_dataset(
        dataset_paths,
        use_gan=True,
        augmentation_factor=augmentation_factor,
        gan_epochs=gan_epochs,
        return_artifacts=True,
        verbose=0,
    )
    return {"without_gan": base, "with_gan": gan}
