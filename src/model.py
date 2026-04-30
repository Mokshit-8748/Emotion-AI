import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, LSTM, Bidirectional, GlobalAveragePooling1D,
    GlobalMaxPooling1D, Multiply, Add, Activation, Layer, SpatialDropout1D,
    Concatenate, MultiHeadAttention, LayerNormalization, LeakyReLU
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

class SEBlock(Layer):
    """Squeeze-and-Excitation Block for channel-wise calibration."""
    def __init__(self, channels, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.gap = GlobalAveragePooling1D()
        self.d1 = Dense(channels // reduction, activation='relu')
        self.d2 = Dense(channels, activation='sigmoid')

    def call(self, x):
        s = self.gap(x)
        s = self.d1(s)
        s = self.d2(s)
        s = tf.reshape(s, [-1, 1, self.channels])
        return x * s

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels, "reduction": self.reduction})
        return config


class ResidualBlock(Layer):
    """Deep Residual Block with SE-calibration."""
    def __init__(self, filters, kernel_size=3, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)
        self.bn1   = BatchNormalization()
        self.conv2 = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)
        self.bn2   = BatchNormalization()
        self.se    = SEBlock(filters)
        self.add   = Add()
        self.shortcut_lyr = None

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.shortcut_lyr = Conv1D(self.filters, 1, padding='same')
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = Activation('relu')(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)
        
        shortcut = inputs
        if self.shortcut_lyr is not None:
            shortcut = self.shortcut_lyr(inputs)
            
        x = self.add([x, shortcut])
        return Activation('relu')(x)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config


class TemporalAttention(Layer):
    """Soft attention over time steps."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = Dense(128, activation='tanh')
        self.output_dense = Dense(1)

    def call(self, x):
        score   = self.score_dense(x)          
        score   = self.output_dense(score)     
        score   = tf.nn.softmax(score, axis=1) 
        context = x * score                    
        return context


def build_model(
    input_shape,
    num_classes,
    learning_rate=3e-4,
    dropout=0.45,
    label_smoothing=0.01,
):
    """
    SOTA SE-ResNet-Transformer Hybrid.
    - Deep Residual Blocks (12+ Convs)
    - Bi-LSTM for sequence modeling
    - Multi-Head Self-Attention (Transformer) for global context
    """
    inputs = Input(shape=input_shape)

    # ── Initial Mapping ──────────────────────────────────────────────
    x = Conv1D(128, 1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # ── Deep Residual Stack ──────────────────────────────────────────
    x = ResidualBlock(128, dilation_rate=1)(x)
    x = ResidualBlock(128, dilation_rate=2)(x)
    x = MaxPooling1D(2)(x)
    x = SpatialDropout1D(0.2)(x)
    
    x = ResidualBlock(256, dilation_rate=1)(x)
    x = ResidualBlock(256, dilation_rate=4)(x)
    x = MaxPooling1D(2)(x)
    x = SpatialDropout1D(0.25)(x)
    
    x = ResidualBlock(256, dilation_rate=1)(x)
    x = ResidualBlock(256, dilation_rate=8)(x)
    x = MaxPooling1D(2)(x)
    x = SpatialDropout1D(0.3)(x)

    # ── Sequence Modeling (Bi-LSTM) ──────────────────────────────────
    # recurrent_dropout=0.001 forces non-CuDNN kernel (required for DirectML)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.001, dropout=0.35))(x)
    x = LayerNormalization()(x)

    # ── Transformer Block (Self-Attention) ───────────────────────────
    mha_out = MultiHeadAttention(num_heads=8, key_dim=32, dropout=0.25)(x, x)
    x = Add()([x, mha_out])
    x = Dropout(0.2)(x)
    x = LayerNormalization()(x)

    # ── Adaptive Feature Fusion ──────────────────────────────────────
    x_att = TemporalAttention()(x)
    avg_p = GlobalAveragePooling1D()(x_att)
    max_p = GlobalMaxPooling1D()(x_att)
    x = Concatenate()([avg_p, max_p])

    # ── Dense Head ───────────────────────────────────────────────────
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(max(0.1, dropout - 0.05))(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=['accuracy'],
    )
    return model
