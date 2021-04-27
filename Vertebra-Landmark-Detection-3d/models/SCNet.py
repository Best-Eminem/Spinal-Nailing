import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class UnetAvgLinear3D(UnetBase):
    """
    U-Net with average pooling and linear upsampling.
    """
    def __init__(self,
                 num_filters_base,
                 repeats=2,
                 dropout_ratio=0.0,
                 kernel_size=None,
                 activation=nn.relu,
                 kernel_initializer=he_initializer,
                 alpha_dropout=False,
                 data_format='channels_first',
                 padding='same',
                 *args, **kwargs):
        super(UnetAvgLinear3D, self).__init__(*args, **kwargs)
        self.num_filters_base = num_filters_base
        self.repeats = repeats
        self.dropout_ratio = dropout_ratio
        self.kernel_size = kernel_size or [3] * 3
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.alpha_dropout = alpha_dropout
        self.data_format = data_format
        self.padding = padding
        self.init_layers()

    def downsample(self, current_level):
        """
        Create and return downsample keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return nn.AvgPool3d([2] * 3)

    def upsample(self, current_level):
        """
        Create and return upsample keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        #return nn.UpSampling3DLinear([2] * 3, data_format=self.data_format)
        return nn.Upsample([2] * 3, mode='trilinear')

    def combine(self, current_level):
        """
        Create and return combine keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return ConcatChannels(data_format=self.data_format)
        # return Add()

    def contracting_block(self, current_level):
        """
        Create and return the contracting block keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        layers = []
        for i in range(self.repeats):
            layers.append(self.conv(current_level, str(i)))
            if self.alpha_dropout:
                layers.append(AlphaDropout(self.dropout_ratio))
            else:
                layers.append(Dropout(self.dropout_ratio))
        return nn.Sequential(layers, name='contracting' + str(current_level))

    def expanding_block(self, current_level):
        """
        Create and return the expanding block keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        layers = []
        for i in range(self.repeats):
            layers.append(self.conv(current_level, str(i)))
            if self.alpha_dropout:
                layers.append(AlphaDropout(self.dropout_ratio))
            else:
                layers.append(Dropout(self.dropout_ratio))
        return nn.Sequential(layers, name='expanding' + str(current_level))

    def conv(self, current_level, postfix):
        """
        Create and return a convolution layer for the current level with the current postfix.
        :param current_level: The current level.
        :param postfix:
        :return:
        """
        return nn.Conv3d(self.num_filters_base,
                      self.kernel_size,
                      name='conv' + postfix,
                      activation=self.activation,
                      data_format=self.data_format,
                      kernel_initializer=self.kernel_initializer,
                      kernel_regularizer=l2(l=1.0),
                      padding=self.padding)
class SpatialConfigurationNet(nn.Module):
    """
    The SpatialConfigurationNet.
    """
    def __init__(self,
                 num_labels,
                 num_filters_base=64,
                 num_levels=4,
                 activation='relu',
                 data_format='channels_first',
                 padding='same',
                 local_activation='none',
                 spatial_activation='none',
                 spatial_downsample=8,
                 dropout_ratio=0.0):
        """
        Initializer.
        :param num_labels: Number of outputs.
        :param num_filters_base: Number of filters for the local appearance and spatial configuration sub-networks.
        :param num_levels: Number of levels for the local appearance and spatial configuration sub-networks.
        :param activation: Activation of the convolution layers ('relu', 'lrelu', or 'selu').
        :param data_format: 'channels_first' or 'channels_last'
        :param padding: Convolution padding.
        :param local_activation: Activation function of local appearance output.
        :param spatial_activation: Activation function of spatial configuration output.
        :param spatial_downsample: Downsample factor for spatial configuration output.
        :param dropout_ratio: The dropout ratio after each convolution layer.
        """
        super(SpatialConfigurationNet, self).__init__()
        self.unet = UnetAvgLinear3D
        self.data_format = data_format
        self.num_filters_base = num_filters_base
        if activation == 'relu':
            activation_fn = nn.relu
            kernel_initializer = he_initializer
            alpha_dropout = False
        elif activation == 'lrelu':
            activation_fn = lambda x: nn.leaky_relu(x, alpha=0.1)
            kernel_initializer = he_initializer
            alpha_dropout = False
        elif activation == 'selu':
            activation_fn = tf.nn.selu
            kernel_initializer = selu_initializer
            alpha_dropout = True
        local_activation_fn, local_heatmap_layer_kernel_initializer = activation_fn_output_kernel_initializer(local_activation)
        spatial_activation_fn, spatial_heatmap_layer_kernel_initializer = activation_fn_output_kernel_initializer(spatial_activation)
        self.downsampling_factor = spatial_downsample
        self.scnet_local = self.unet(num_filters_base=self.num_filters_base, num_levels=num_levels, kernel_initializer=kernel_initializer, alpha_dropout=alpha_dropout, activation=activation_fn, dropout_ratio=dropout_ratio, data_format=data_format, padding=padding)
        self.local_heatmaps = nn.Sequential([Conv3D(num_labels, [1] * 3, name='local_heatmaps', kernel_initializer=local_heatmap_layer_kernel_initializer, activation=None, data_format=data_format, padding=padding),
                                          Activation(local_activation_fn, dtype='float32', name='local_heatmaps')])
        self.downsampling = AveragePooling3D([self.downsampling_factor] * 3, name='local_downsampling', data_format=data_format)
        self.scnet_spatial = self.unet(num_filters_base=self.num_filters_base, num_levels=num_levels, repeats=1, kernel_initializer=kernel_initializer, activation=activation_fn, alpha_dropout=alpha_dropout, dropout_ratio=dropout_ratio, data_format=data_format, padding=padding)
        self.spatial_heatmaps = Conv3D(num_labels, [1] * 3, name='spatial_heatmaps', kernel_initializer=spatial_heatmap_layer_kernel_initializer, activation=None, data_format=data_format, padding=padding)
        self.upsampling = Sequential([UpSampling3DCubic([self.downsampling_factor] * 3, name='spatial_upsampling', data_format=data_format),
                                      Activation(spatial_activation_fn, dtype='float32', name='spatial_heatmaps')])

    def call(self, inputs, training, **kwargs):
        """
        Call model.
        :param inputs: Input tensors.
        :param training: If True, use training mode, otherwise testing mode.
        :param kwargs: Not used.
        :return: (heatmaps, local_heatmaps, spatial_heatmaps) tuple.
        """
        node = self.scnet_local(inputs, training=training)
        local_heatmaps = node = self.local_heatmaps(node, training=training)
        node = self.downsampling(node, training=training)
        node = self.scnet_spatial(node, training=training)
        node = self.spatial_heatmaps(node, training=training)
        spatial_heatmaps = self.upsampling(node, training=training)
        heatmaps = local_heatmaps * spatial_heatmaps

        return heatmaps, local_heatmaps, spatial_heatmaps