import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, losses
from deepsleep.nn import *
from deepsleep.loss import focal_loss

class DeepFeatureNet():

    def __init__(
        self, 
        batch_size, 
        input_dims, 
        n_classes, 
        is_train, 
        reuse,
        name="deepfeaturenet"
    ):
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.n_classes = n_classes
        self.is_train = is_train
        self.reuse = reuse
        self.name = name
        self.activations = []
        self.layer_idx = 1

    def _build_placeholder(self):
        # Input
        name = "x_train" if self.is_train else "x_valid"
        self.input_var = tf.compat.v1.placeholder(
            tf.float32, 
            shape=[self.batch_size, self.input_dims, 1, 1],
            name=name + "_inputs"
        )
        # Target
        self.target_var = tf.compat.v1.placeholder(
            tf.int32, 
            shape=[self.batch_size, ],
            name=name + "_targets"
        )

    def _conv1d_layer(self, input_var, filter_size, n_filters, stride, wd=0):
        name = "l{}_conv".format(self.layer_idx)
        output = layers.Conv2D(n_filters, (filter_size, 1), (stride, 1), padding='SAME',
                               kernel_regularizer=regularizers.l2(wd), name=name)(input_var)
        output = layers.BatchNormalization(axis=-1, momentum=0.999, epsilon=1e-5)(output)
        output = tf.nn.relu(output)
        self.activations.append((name, output))
        self.layer_idx += 1
        return output
       
    def build_cnn_model(self, input_var, reuse):
        # List to store the output of each CNNs
        output_conns_1 = []
        
        ######### CNNs with small filter size at the first layer #########       
        network = self._conv1d_layer(input_var=input_var, 
                                     filter_size=64, 
                                     n_filters=128, 
                                     stride=6, 
                                     wd=1e-3)

        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = layers.MaxPooling2D(pool_size=(8, 1),strides=(8, 1),padding='same')(network)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        name = "l{}_dropout".format(self.layer_idx)
        if self.is_train:
            network = layers.Dropout(0.5)(network)
        else:
            network = layers.Dropout(0)(network)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Convolution
        network = self._conv1d_layer(input_var=network, 
                                     filter_size=6, 
                                     n_filters=128, 
                                     stride=1)
        network = self._conv1d_layer(input_var=network, 
                                     filter_size=6, 
                                     n_filters=128, 
                                     stride=1)
        network = self._conv1d_layer(input_var=network, 
                                     filter_size=6, 
                                     n_filters=128, 
                                     stride=1)
        
        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = layers.MaxPooling2D(pool_size=(4, 1),strides=(4, 1),padding='same')(network)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Flatten
        name = "l{}_flat".format(self.layer_idx)
        network = layers.Flatten()(network)
        self.activations.append((name, network))
        self.layer_idx += 1

        output_conns_1.append(network)
 
        ######### CNNs with medium filter size at the first layer #########

        # Convolution
        network = self._conv1d_layer(input_var=input_var, 
                                     filter_size=640, 
                                     n_filters=128, 
                                     stride=64,
                                     wd=1e-3)

        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = layers.MaxPooling2D(pool_size=(6, 1),strides=(6, 1),padding='same')(network)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        name = "l{}_dropout".format(self.layer_idx)
        if self.is_train:
            network = layers.Dropout(0.5)(network)
        else:
            network = layers.Dropout(0)(network)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Convolution
        network = self._conv1d_layer(input_var=network, 
                                     filter_size=10, 
                                     n_filters=128, 
                                     stride=1)
        network = self._conv1d_layer(input_var=network, 
                                     filter_size=10, 
                                     n_filters=128, 
                                     stride=1)
        network = self._conv1d_layer(input_var=network, 
                                     filter_size=10, 
                                     n_filters=128, 
                                     stride=1)
        
        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = layers.MaxPooling2D(pool_size=(2, 1),strides=(2, 1),padding='same')(network)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Flatten
        name = "l{}_flat".format(self.layer_idx)
        network = layers.Flatten()(network)
        self.activations.append((name, network))
        self.layer_idx += 1

        output_conns_1.append(network)
        
        ######### Aggregate and link two CNNs#########
        # Concat
        name = "l{}_concat".format(self.layer_idx)
        network = tf.concat(output_conns_1, 1, name=name)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        name = "l{}_dropout".format(self.layer_idx)
        if self.is_train:
            network = layers.Dropout(0.3)(network)
        else:
            network = layers.Dropout(0)(network)
        self.activations.append((name, network))
        self.layer_idx += 1

        name = "l{}_fc".format(self.layer_idx)
        network = layers.Dense(800, name=name)(network)
        network = layers.BatchNormalization(axis=-1, momentum=0.999, epsilon=1e-5)(network)
        network = tf.nn.relu(network)
        self.activations.append((name, network))
        self.layer_idx += 1
           
        # Dropout
        name = "l{}_dropout".format(self.layer_idx)
        if self.is_train:
            network = layers.Dropout(0.5)(network)
        else:
            network = layers.Dropout(0)(network)
        self.activations.append((name, network))
        self.layer_idx += 1
        
        name = "l{}_fc".format(self.layer_idx)
        network = layers.Dense(400, name=name)(network)
        network = layers.BatchNormalization(axis=-1, momentum=0.999, epsilon=1e-5)(network)
        network = tf.nn.relu(network)
        self.activations.append((name, network))
        self.layer_idx += 1
        
        return network

    def init_ops(self):
        self._build_placeholder()

        # Get loss and prediction operations
        with tf.compat.v1.variable_scope(self.name) as scope:
            
            # Reuse variables for validation
            if self.reuse:
                scope.reuse_variables()

            # Build model
            network = self.build_cnn_model(input_var=self.input_var, reuse=self.reuse)

            # Softmax linear
            name = "l{}_softmax_linear".format(self.layer_idx)
            network = layers.Dense(self.n_classes, name=name)(network)
            self.activations.append((name, network))
            self.layer_idx += 1

            ######### Compute loss #########

            # Cross-entropy loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.target_var,
                logits=network,
                name="sparse_softmax_cross_entropy_with_logits"
            )
            self.loss_op = tf.reduce_mean(loss, name="cross_entropy")
            
#            # Regularization loss
#            regular_loss = tf.add_n(
#                tf.compat.v1.get_collection("losses", scope=scope.name + "\/"),
#                name="regular_loss"
#            )

#            # Total loss
#            self.loss_op = tf.add(loss, regular_loss)
            
            # Predictions
            self.pred_op = tf.argmax(network, axis=1)

class DeepSleepNet(DeepFeatureNet):

    def __init__(
        self, 
        batch_size, 
        input_dims, 
        n_classes, 
        seq_length,
        n_rnn_layers,
        is_train, 
        reuse,
        name="deepsleepnet"
    ):
        super().__init__(
            batch_size=batch_size, 
            input_dims=input_dims, 
            n_classes=n_classes, 
            is_train=is_train, 
            reuse=reuse, 
            name=name
        )

        self.seq_length = seq_length
        self.n_rnn_layers = n_rnn_layers

    def _build_placeholder(self):
        # Input
        name = "x_train" if self.is_train else "x_valid"
        self.input_var = tf.compat.v1.placeholder(
            tf.float32, 
            shape=[self.batch_size*self.seq_length, self.input_dims, 1, 1],
            name=name + "_inputs"
        )

        # Target
        self.target_var = tf.compat.v1.placeholder(
            tf.int32, 
            shape=[self.batch_size*self.seq_length, ],
            name=name + "_targets"
        )

    def build_GRU_model(self, input_var, reuse):
        
        output_conns = []      
        network = super().build_cnn_model(input_var=self.input_var, reuse=self.reuse)              
        output_conns.append(network)
        res = network
        ######################################################################
        name = "l{}_reshape_seq".format(self.layer_idx)
        input_dim = network.get_shape()[-1]
        seq_input = tf.reshape(network, shape=[-1, self.seq_length, input_dim], name=name)
        assert self.batch_size == seq_input.get_shape()[0]
        self.activations.append((name, seq_input))
        self.layer_idx += 1

        # Bidirectional GRU network
        name = "l{}_bi_GRU".format(self.layer_idx)
        hidden_size = 200
        network = layers.Bidirectional((layers.GRU(hidden_size, dropout=0.5,
                                                   recurrent_dropout=0.5, 
                                                   return_sequences=True,
                                                   input_shape=seq_input.get_shape(),
                                                   name=name)))(seq_input)
        network = tf.reshape(network, shape=[-1, hidden_size*2])            
        self.activations.append((name, network))
        self.layer_idx +=1

        # Append output
        output_conns.append(network)
        ######################################################################

        # Add
        name = "l{}_add".format(self.layer_idx)
        network = tf.add(res, network, name=name)
#        network = tf.concat(output_conns, 1, name=name)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        name = "l{}_dropout".format(self.layer_idx)
        if self.is_train:
            network = layers.Dropout(0.5)(network)
        else:
            network = layers.Dropout(0)(network)
        self.activations.append((name, network))
        self.layer_idx += 1
         
        name = "l{}_fc".format(self.layer_idx)
        network = layers.Dense(200, name=name)(network)
        network = layers.BatchNormalization(axis=-1, momentum=0.999, epsilon=1e-5)(network)
        network = tf.nn.relu(network)
        self.activations.append((name, network))
        self.layer_idx += 1

        return network

    def init_ops(self):
        self._build_placeholder()

        # Get loss and prediction operations
        with tf.compat.v1.variable_scope(self.name) as scope:
            
            # Reuse variables for validation
            if self.reuse:
                scope.reuse_variables()

            # Build model
            network = self.build_GRU_model(input_var=self.input_var, reuse=self.reuse)

            # Softmax linear
            name = "l{}_softmax_linear".format(self.layer_idx)
            network = fc(name=name, input_var=network, 
                         n_hiddens=self.n_classes, 
                         bias=0.0, 
                         wd=0)
            self.activations.append((name, network))
            self.layer_idx += 1

            # Outputs of softmax linear are logits
            self.logits = network

            ######### Compute loss #########

            # Focal cross-entropy loss for a sequence of logits           
            softed_output = tf.nn.softmax(self.logits)
            loss = focal_loss(self.target_var, softed_output, gamma=2.0, alpha=0.25)
            
            loss = tf.reduce_sum(input_tensor=loss)/self.batch_size

            # Regularization loss
            regular_loss = tf.add_n(
                tf.compat.v1.get_collection("losses", scope=scope.name + "\/"),
                name="regular_loss"
            )

            # Total loss
            self.loss_op = tf.add(loss, regular_loss)

            # Predictions
            self.pred_op = tf.argmax(input=self.logits, axis=1)
