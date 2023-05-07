import os
import tensorflow as tf
import numpy as np
import time
# import tf_slim
from datetime import datetime
from tensorflow.keras import layers, models, Input, regularizers
from deepsleep.data_loader_multich2 import NonSeqDataLoader, SeqDataLoader
from deepsleep.utils import iterate_minibatches, iterate_batch_seq_minibatches
from sklearn.metrics import confusion_matrix, f1_score

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if physical_devices:
#  tf.config.experimental.set_memory_growth(physical_devices[0], True8 )
'''
tf.enable_eager_execution(
    config=None,
    device_policy=None,
    execution_mode=None
)
from keras import backend as K
K.clear_session()
'''

time_start = time.time()  # 记录开始时间

total_cm = np.zeros((5,5))
val_cm = np.zeros((5,5))
for numm in range(5,6,1):
    ii = str(numm)
    data_dir = 'E:/data2/'+ii
    n_folds=2
    fold_idx=-1
    resume=False
    n_classes=5
    EPOCH_SEC_LEN=30
    SAMPLING_RATE=128
    seq_length=20
    hidden_size_GRU=128
    seq_length_GRU=20
    
    batch_size = 32
    input_dims = SAMPLING_RATE*EPOCH_SEC_LEN

    
    def print_performance(n_train_examples, n_valid_examples, train_cm, valid_cm, 
                          epoch, n_epochs, train_duration, train_loss, train_acc, 
                          train_f1, valid_duration, valid_loss, valid_acc, valid_f1):
        # Print performance
        if ((epoch) % 10 == 0) or ((epoch) == n_epochs):
            print (" ")
            print ("[{}] epoch {}:".format(datetime.now(), epoch))
            print ("train ({:.3f} sec): n={}, loss={:.3f}, acc={:.3f}, "
                   "f1={:.3f}".format(train_duration, n_train_examples,
                    train_loss, train_acc, train_f1))
            print (train_cm)
            print ("valid ({:.3f} sec): n={}, loss={:.3f}, acc={:.3f}, "
                "f1={:.3f}".format(valid_duration, n_valid_examples,
                    valid_loss, valid_acc, valid_f1))
            print (valid_cm)
            print (" ")
        else:
            print ("epoch {}: train ({:.2f} sec): n={}, loss={:.3f}, "
                "acc={:.3f}, f1={:.3f} | "
                "valid ({:.2f} sec): n={}, loss={:.3f}, "
                "acc={:.3f}, f1={:.3f}".format(epoch, train_duration, 
                     n_train_examples, train_loss, train_acc, train_f1, valid_duration, 
                     n_valid_examples, valid_loss, valid_acc, valid_f1))
                    
            
    class Conv1D_Block(tf.keras.Model):
      def __init__(self, filter_size, n_filters, stride, wd=0):
        super(Conv1D_Block, self).__init__(name='')
        
        self.conv_1d = layers.Conv2D(n_filters, (2, filter_size), (1, stride), 
                                     padding='SAME', kernel_regularizer=regularizers.l2(wd))
        self.bn = layers.BatchNormalization(axis=-1, momentum=0.999, epsilon=1e-5)
       
      def call(self, input_tensor, training=False):
        x = self.conv_1d(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        
        return x
    
    class Dense_Block(tf.keras.Model):
      def __init__(self, hidden_size):
        super(Dense_Block, self).__init__(name='')
        
        self.Dense = layers.Dense(hidden_size)
        self.bn = layers.BatchNormalization(axis=-1, momentum=0.999, epsilon=1e-5)
       
      def call(self, input_tensor, training=False):
        x = self.Dense(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        
        return x
    
    class SELayer(tf.keras.Model):
        def __init__(self, channel, reduction=16):
            super(SELayer, self).__init__()
            self.fc1 =  layers.Dense(channel//reduction, activation = None)
            self.fc3 =  layers.Dense(channel, activation = None)
    
        def call(self, x):
            shape1 = x.shape
            _,b,c,d = shape1.as_list() #每一个尺度cnn后加se时候用这一行，反之下一行
            # b,c = shape1.as_list()
            y = tf.reduce_mean(x,[1,2]) #每一个尺度cnn后加se时候用这一行，反之屏蔽 each_c1 是1，2都reduce了
            # y = x  #合并后加se用这一行，反之屏蔽
            y = self.fc1(y)
            y = tf.nn.relu(y)
            y = self.fc3(y)
            y = tf.sigmoid(y)
            y = tf.reshape(y,[-1,1,1,d])#每一个尺度cnn后加se时候用这一行，反之下一行
            # y = tf.reshape(y,[-1,c])
            return x * y
    
    class Attention(tf.keras.layers.Layer): #多头注意力
        '''
    hidden_size=512,  # 隐藏层神经元数量（全连接层的第二层）
    num_heads=8,  # 多头注意力机制中head数量
    attention_dropout=0.1, # 多头注意力机制中dropout参数
        '''
        def __init__(self,hidden_size,num_heads,attention_dropout,train):
            # hidden 必须能与 num_head 整除
            if hidden_size % num_heads != 0:
                raise ValueError('Hidden size must be evenly divisible by the number of ""heads')
            super(Attention,self).__init__()
            self.hidden_size=hidden_size
            self.num_heads=num_heads
            self.attention_dropout=attention_dropout
            self.train=train
    
            # 计算'q','k','v'
            self.q_dense_layer=tf.keras.layers.Dense(hidden_size,use_bias=False,name='q')
            self.k_dense_layer=tf.keras.layers.Dense(hidden_size,use_bias=False,name='k')
            self.v_dense_layer=tf.keras.layers.Dense(hidden_size,use_bias=False,name='v')
    
            # attention输出层
            self.output_dense_layer=tf.keras.layers.Dense(hidden_size,use_bias=False,name='outpout_dropout')
        def split_heads(self,x):
            """
            将x拆分不同的注意力head，并将结果转置(转置的目的是为了矩阵相乘时维度正确)
            :param x: shape[batch_size,length,hidden_size]
            :return: shape[batch_size,num_heads,length,hidden_size/num_heads]
            """
    
            with tf.name_scope('split_heads'):
                batch_size=x.get_shape()[0]
                length=x.get_shape()[1]
                # print(shape)
                # print(batch_size)
                # print(length)
    # 
                # 计算最后一个维度的深度
                depth=(self.hidden_size // self.num_heads)
    
                # 拆分最后一个维度
                x=tf.reshape(x,[-1,length,self.num_heads,depth])
                # print(x.shape)
    
                # 将结果转置,即：[batch_size,self.num_heads,length,depth]
                return tf.transpose(x,[0,2,1,3])
    
        def combine_heads(self,x):
            """
            将拆分的张量再次连接(split_heads逆操作),input是split_heads_fn的输出
            :param x: shape[batch_size,num_heads,length,hidden_size/num_heads]
            :return:  shape[batch_size,length,hidden_size]
            """
            with tf.name_scope('combine_heads'):
                batchs_size=x.get_shape()[0]
                length=x.get_shape()[2]
    
                # [batch_size,length,num_heads,depth]
                x=tf.transpose(x,[0,2,1,3])
                return tf.reshape(x,[-1,length,self.hidden_size])
    
    
        def call(self,x,bias=0,cache=None):
            """
    
            :param x: shape[batch_size,length_x,hidden_size]
            :param y: shape[batch_size,length_y,hidden_size]
            :param bias: 与点积结果相加
            :param cache: 预测模式使用；返回类型为字典：
            {
            'k':shape[batch_size,i,key_channels],
            'v':shape[batch_size,i,value_channels]
            }
            i:当前decoded长度
            :return: shape[batch_size,length_x,hidden_size]
            """
            # 获取'q','k','v'

            q=self.q_dense_layer(x)
            k=self.k_dense_layer(x)
            v=self.v_dense_layer(x)
            x_q = q
            # print(k.shape)
            # print(v.shape)
            # 预测模式
            if cache is not None:
                # 合并k和v值
                k=tf.concat([cache['k'],k],axis=1)
                v=tf.concat([cache['v'],v],axis=1)
    
                cache['k']=k
                cache['v']=v
            # 将q,k,v拆分
            q=self.split_heads(q)
            k=self.split_heads(k)
            v=self.split_heads(v)
    
            #缩放q以防止q和k之间的点积过大
            depth = (self.hidden_size // self.num_heads)
            q *= depth ** -0.5
    
            # 计算点积,将k转置
            logits=tf.matmul(q,k,transpose_b=True)
            logits+=bias
            weights=tf.nn.softmax(logits,name='attention_weight')
    
            # 训练模式使用dropout
            if self.train:
                weights=tf.nn.dropout(weights,1.0-self.attention_dropout)
            attention_outpout=tf.matmul(weights,v)
    
            # 单头结束，计算多头
            attention_outpout=self.combine_heads(attention_outpout)
    
            # 使用全连接层输出
            attention_outpout=self.output_dense_layer(attention_outpout)
    
            return attention_outpout, x_q
    
    class FeedFowardNetWork(tf.keras.layers.Layer):
        """
        全连接层,共2层
        hidden_size=512,  # 隐藏层神经元数量（全连接层的第二层）
        filter_size=2048,  # feedforward连接层中神经元数量
        relu_dropout=0.1, # 全连接层中dropout设置
        """
        def __init__(self,hidden_size,filter_size,relu_dropout,train,allow_pad):
            super(FeedFowardNetWork,self).__init__()
    
            self.hidden_size=hidden_size
            self.filter_size=filter_size
            self.relu_dropout=relu_dropout
            self.train=train
    
            # 模型默认需要固定长度
            self.all_pad=allow_pad
    
            self.filter_dense_layer=tf.keras.layers.Dense(
                filter_size,use_bias=True,activation=tf.nn.relu,
                name='filter_layer'
            )
            self.outpout_dense_layer=tf.keras.layers.Dense(
                hidden_size,use_bias=True,name='outpout_layer'
            )
        def call(self,x,padding=None):
            """
            返回全连接层输出
            :param x: shape[batch_size,length,hidden_size]
            :param padding:shape[batch_size,length]
            :return:
            """
            padding=None if not self.all_pad else padding
    
            # 获取已知shape
            batch_size=x.get_shape()[0]
            length=x.get_shape()[1]
    
            if padding is not None:
                with tf.name_scope('remove_padding'):
                    pad_mask=tf.reshape(padding,[-1])
                    nopad_ids=tf.to_int32(tf.where(pad_mask<1e-9))
    
                    # 将x维度修改成[batch_size,selt.hidden_size]以移除padding
                    x=tf.reshape(x[-1,self.hidden_size])
                    x=tf.gather_nd(x,indices=nopad_ids)
    
                    # 扩展一维
                    x.set_shape([None, self.hidden_size])
                    x = tf.expand_dims(x, axis=0)
            outpout=self.filter_dense_layer(x)
    
            # 训练模式使用dropout
            if self.train:
                outpout=tf.nn.dropout(outpout,1.0-self.relu_dropout)
            outpout=self.outpout_dense_layer(outpout)
    
            if padding is not None:
                with tf.name_scope('re_add_padding'):
                    # 去除指定维度中，大小为1的
                    output=tf.squeeze(outpout,axis=0)
                    output = tf.scatter_nd(
                        indices=nopad_ids,
                        updates=output,
                        shape=[batch_size * length, self.hidden_size]
                    )
                    output = tf.reshape(output, [-1, length, self.hidden_size])
                    
                return output
            
            return outpout
    
    class LayerNormalization(tf.keras.layers.Layer):
        def __init__(self, epsilon=1e-6, **kwargs):
            self.eps = epsilon
            super(LayerNormalization, self).__init__(**kwargs)
        def build(self, input_shape):
            self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                         initializer=tf.ones_initializer(), trainable=True)
            self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                        initializer=tf.zeros_initializer(), trainable=True)
            super(LayerNormalization, self).build(input_shape)
        def call(self, x):
            mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
            std = tf.keras.backend.std(x, axis=-1, keepdims=True)
            return self.gamma * (x - mean) / (std + self.eps) + self.beta
        def compute_output_shape(self, input_shape):
            return input_shape
    
    def feature_learning_model(input_dims, n_classes):
        x = Input(shape=(seq_length, input_dims, 1), dtype='float32', name='Input')
        ######### CNNs with detail filter size ######### 
        x_1 = Conv1D_Block(64, 128, 6, 1e-3)(x)
        x_1 = layers.MaxPooling2D((1, 8),(1, 8))(x_1)
        x_1 = layers.Dropout(0.5)(x_1)
        x_1 = Conv1D_Block(6, 128, 1)(x_1)
        x_1 = Conv1D_Block(6, 128, 1)(x_1)
        x_1 = Conv1D_Block(6, 128, 1)(x_1)
        x_1 = Conv1D_Block(6, 128, 1)(x_1)
        x_1 = layers.MaxPooling2D((1, 4),(1, 4))(x_1)
        # print(x_1.shape)
        # print("###")
        x_1 = SELayer(128,reduction=16)(x_1) 
        shape1 = x_1.get_shape()
        x_1 = tf.reshape(x_1, [-1, shape1[1] , shape1[2] * shape1[3]])  
                  
        ######### CNNs with shape filter size #########    
        x_2 = Conv1D_Block(640, 128, 64, 1e-3)(x)
        x_2 = layers.MaxPooling2D((1, 6),(1, 6))(x_2)
        x_2 = layers.Dropout(0.5)(x_2)
        x_2 = Conv1D_Block(10, 128, 1)(x_2)
        x_2 = Conv1D_Block(10, 128, 1)(x_2)
        x_2 = Conv1D_Block(10, 128, 1)(x_2)
        x_2 = Conv1D_Block(10, 128, 1)(x_2)
        x_2 = layers.MaxPooling2D((1, 2),(1, 2))(x_2)
        x_2 = SELayer(128,reduction=16)(x_2)  
        shape2 = x_2.get_shape()
        x_2 = tf.reshape(x_2, [-1, shape2[1] , shape2[2] * shape2[3]])     

        
        ######### concatenate and filter ######### 
        x_c = layers.concatenate([x_1, x_2], axis=2)
        x_c = layers.Dropout(0.5)(x_c)
        x_c = Dense_Block(512)(x_c)
        x_c = layers.Dropout(0.5)(x_c)
        x_c = Dense_Block(256)(x_c)
        # x_c = SELayer(400, reduction=16)(x_c)
        model = models.Model(x, x_c)
        return model
    
    f_1_model = feature_learning_model(input_dims, n_classes)
    # f_1_model.summary()
    
    def feature_learning_lastdense(n_classes):
        x = Input(shape=(seq_length, 256), dtype='float32', name='Input')
        output = layers.Dense(n_classes, activation='softmax')(x)
        model = models.Model(x, output)
        return model
        
    f_2_model = feature_learning_lastdense(n_classes)
    # f_2_model.summary()
    
    f_model = models.Sequential()   
    f_model.add(f_1_model)
    f_model.add(f_2_model)
    f_model.load_weights('E:/Lab/EOG_Code/infant_sleep - tf20-2/network/model/multi-group/CNN_ATTN_Ts_ADDLOSS/MASS/C4/20/fold'
                          +ii+'/feature_learning/f_model_weights.h5')
    ###########################################################    
    
    def sequence_learning_model(seq_length, hidden_size_GRU, n_classes):
        x_0 = Input(shape=(seq_length, 256), dtype='float32', name='Input')
        x_attn, x_q = Attention(hidden_size=128, num_heads=8, attention_dropout=0.1, train = True)(x_0)
        x_add1 = tf.add(x_attn, x_q)
        x_ffd = FeedFowardNetWork(128, 512, 0.1, train = True, allow_pad = True)(x_add1)
        x_add = tf.add(x_attn, x_ffd)
        x_out = LayerNormalization()(x_add)  
        
        ######### concatenate and filter ######### 
        # x_2 = tf.add(x_0, x_out)
        x_2 = layers.Dropout(0.5)(x_out)
        # x_2 = Dense_Block(128)(x_2)
        output = layers.Dense(n_classes, activation='softmax')(x_2)
        model = models.Model(x_0, output)
        return model
    
    s_1_model = sequence_learning_model(seq_length_GRU, hidden_size_GRU, n_classes)
    # s_1_model.summary()
    
    s_model = models.Sequential()
    s_model.add(f_1_model)
    s_model.add(s_1_model)
    # s_model.summary()    
    s_model.load_weights('E:/Lab/EOG_Code/infant_sleep - tf20-2/network/model/multi-group/CNN_ATTN_Ts_ADDLOSS/MASS/C4/20/fold'
                          +ii+'/sequence_learning/s_model_weights.h5')

    
    ##########################################################
    
    @tf.function
    def f_train_step(inputs, labels, training=True):
        with tf.GradientTape() as tape:
            pred = f_model(inputs, training=training)
            regular_loss = tf.math.add_n(f_model.losses)
            pred_loss = loss_fn(labels, pred)
            total_loss = pred_loss + regular_loss
    
        gradients = tape.gradient(total_loss, f_model.trainable_variables)
        optimizer_f.apply_gradients(zip(gradients, f_model.trainable_variables))
        
        pred_y = tf.argmax(pred, axis=2)
        return pred_y, total_loss, pred
    
    def f_valid_step(inputs, labels, training=False):
        pred = f_model(inputs, training=training)
        regular_loss = tf.math.add_n(f_model.losses)
        pred_loss = loss_fn(labels, pred)
        total_loss = pred_loss + regular_loss
        pred_y = tf.argmax(pred, axis=2)
        return pred_y, total_loss, pred
    
    def f_run_epoch(inputs, targets, batch_size, training=True):
        start_time = time.time()
        y = []
        y_prob = []
        y_true = []
        total_loss, n_batches = 0.0, 0
        for x_batch, y_batch in iterate_minibatches(inputs, targets, 
                                                    batch_size, shuffle=False):
            _,seq_length,input_dims,dims = x_batch.shape
            # x_batch = x_batch.reshape(batch_size*seq_length,input_dims,dims)
            # x_batch = x_batch[:,:,:,np.newaxis]
            # y_batch = y_batch.reshape(batch_size*seq_length)
            if training == True:
                y_pred, loss_value, y_predprobility = f_train_step(x_batch, y_batch, training=training)           
            else:
                y_pred, loss_value, y_predprobility = f_valid_step(x_batch, y_batch, training=training)     
            
            total_loss += loss_value
            n_batches += 1
            y_pred = tf.reshape(y_pred, [batch_size, seq_length])
            y_batch = y_batch.reshape(batch_size, seq_length)
            y_predprobility = tf.reshape(y_predprobility, [batch_size, seq_length, -1])
            if len(y) == 0:
                y = y_pred
                y_true = y_batch
                y_prob = y_predprobility
            else:
                y = np.append(y,y_pred,axis=0)
                y_true = np.append(y_true, y_batch, axis=0)
                y_prob = np.append(y_prob, y_predprobility, axis=0)
            
            
            # Check the loss value
            assert not np.isnan(loss_value), \
            "Model diverged with loss = NaN"
    
        duration = time.time() - start_time
        total_loss /= n_batches
        total_y_pred = y
        total_y_true = y_true
        total_y_prob = y_prob
            
        return total_y_true, total_y_pred, total_loss, duration, total_y_prob
    
    def s_train_step(inputs, labels, training=True):
        with tf.GradientTape(persistent=True) as tape:
            pred1 = f_model(inputs, training=training)
            regular_loss1 = tf.math.add_n(f_model.losses)
            pred_loss1 = loss_fn(labels, pred1)
            
            pred = s_model(inputs, training=training)
            regular_loss = tf.math.add_n(s_model.losses)
            pred_loss = loss_fn(labels, pred)
            total_loss = pred_loss + regular_loss + pred_loss1 + regular_loss1
    
        gradients_1 = tape.gradient(total_loss, f_1_model.trainable_variables)
        optimizer_1.apply_gradients(zip(gradients_1, f_1_model.trainable_variables))
        gradients_2 = tape.gradient(total_loss, s_1_model.trainable_variables)
        optimizer_2.apply_gradients(zip(gradients_2, s_1_model.trainable_variables))
        
        pred_y = tf.argmax(pred, axis=2)
        return pred_y, total_loss, pred
    
    def s_valid_step(inputs, labels, training=False):
        pred = s_model(inputs, training=training)
        regular_loss = tf.math.add_n(s_model.losses)
        pred_loss = loss_fn(labels, pred)
        total_loss = pred_loss + regular_loss
        pred_y = tf.argmax(pred, axis=2)
        return pred_y, total_loss, pred
    
    def s_run_epoch(inputs, targets, batch_size, training=True):
        start_time = time.time()
        y = []
        y_prob = []
        y_true = []
        total_loss, n_batches = 0.0, 0
        for x_batch, y_batch in iterate_minibatches(inputs, targets, 
                                                    batch_size, shuffle=False):
            _,seq_length,input_dims,dims = x_batch.shape
            # x_batch = x_batch.reshape(batch_size*seq_length,input_dims,dims)
            # x_batch = x_batch[:,:,:,np.newaxis]
            # y_batch = y_batch.reshape(batch_size*seq_length)
            if training == True:
                y_pred, loss_value, y_predprobility = s_train_step(x_batch, y_batch, training=training)           
            else:
                y_pred, loss_value, y_predprobility = s_valid_step(x_batch, y_batch, training=training)     
            
            total_loss += loss_value
            n_batches += 1
            y_pred = tf.reshape(y_pred, [batch_size, seq_length])
            y_batch = y_batch.reshape(batch_size, seq_length)
            y_predprobility = tf.reshape(y_predprobility, [batch_size, seq_length, -1])
            if len(y) == 0:
                y = y_pred
                y_true = y_batch
                y_prob = y_predprobility
            else:
                y = np.append(y,y_pred,axis=0)
                y_true = np.append(y_true, y_batch, axis=0)
                y_prob = np.append(y_prob, y_predprobility, axis=0)
            
            
            # Check the loss value
            assert not np.isnan(loss_value), \
            "Model diverged with loss = NaN"
    
        duration = time.time() - start_time
        total_loss /= n_batches
        total_y_pred = y
        total_y_true = y_true
        total_y_prob = y_prob
            
        return total_y_true, total_y_pred, total_loss, duration, total_y_prob
    
   
    
    # Make subdirectory for pretraining
    
    optimizer_f = tf.keras.optimizers.Adam(0)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    step_f = tf.Variable(1, name='step', trainable=False)
    ckpt_f = tf.train.Checkpoint(step=step_f, optimizer=optimizer_f, net=f_model)

    
    optimizer_1 = tf.keras.optimizers.Adam(0)
    optimizer_2 = tf.keras.optimizers.Adam(0)
    loss_s = tf.keras.losses.SparseCategoricalCrossentropy()
    

    data_loader_1 = NonSeqDataLoader(data_dir, n_folds, fold_idx, n_classes,seq_length)
    x_train, y_train, x_valid, y_valid = data_loader_1.load_train_data() #这里出来就是（N-19）x20x3840x2的数据
    
    validaccmax = 0
    
    y_true_train, y_pred_train, train_loss, train_duration, y_prob_train = \
                                s_run_epoch(x_train, y_train, batch_size, training=False)
                                
    y_len_train = len(y_true_train)
    # print(y_true_train[0:10,:])
    y_nonseq_train = np.zeros((y_len_train + seq_length - 1))
    for nn in range(seq_length-1):
        y_nonseq_train[nn] = y_true_train[0,nn]
    for nnn in range(y_len_train):
        y_nonseq_train[nnn+seq_length-1] = y_true_train[nnn,seq_length-1]
    
    temp_y_prob_train = np.ones((y_len_train + seq_length - 1, seq_length, n_classes))
    for nnnn in range(seq_length):
        temp_y_prob_train[nnnn:(nnnn+y_len_train),nnnn,:] = y_prob_train[:,nnnn,:]
    # temp_y_prob_train += 1e-8
    temp_y_prob_train = np.log(temp_y_prob_train)
    y_prob_train_sum = np.sum(temp_y_prob_train, axis=1)
    y_nonseq_train_pred = tf.argmax(y_prob_train_sum, axis=1)
    y_nonseq_train_pred.numpy()
    
    y_true_train = y_true_train.reshape(-1,1)
    y_pred_train = y_pred_train.reshape(-1,1)
    
    # print(y_nonseq_train_pred)
    
    
    n_train_examples = len(y_nonseq_train)
    train_cm = confusion_matrix(y_nonseq_train, y_nonseq_train_pred)
    train_acc = np.mean(y_nonseq_train == y_nonseq_train_pred)
    train_f1 = f1_score(y_nonseq_train, y_nonseq_train_pred, average="macro") 

    y_true_val, y_pred_val, valid_loss, valid_duration, y_prob_val = \
                            s_run_epoch(x_valid, y_valid, batch_size, training=False)
                            
    y_len_val = len(y_true_val)
    y_nonseq_val = np.zeros((y_len_val + seq_length - 1))
    for nn in range(seq_length-1):
        y_nonseq_val[nn] = y_true_val[0,nn]
    for nnn in range(y_len_val):
        y_nonseq_val[nnn+seq_length-1] = y_true_val[nnn,seq_length-1]  
    
    temp_y_prob_val = np.ones((y_len_val + seq_length - 1, seq_length, n_classes))
    for nnnn in range(seq_length):
        temp_y_prob_val[nnnn:(nnnn+y_len_val),nnnn,:] = y_prob_val[:,nnnn,:]
    temp_y_prob_val = np.log(temp_y_prob_val)
    y_prob_val_sum = np.sum(temp_y_prob_val, axis=1)
    y_nonseq_val_pred = tf.argmax(y_prob_val_sum, axis=1)
    y_nonseq_val_pred.numpy()
    
    y_true_val = y_true_val.reshape(-1,1)
    y_pred_val = y_pred_val.reshape(-1,1)
    
    n_valid_examples = len(y_nonseq_val)
    valid_cm = confusion_matrix(y_nonseq_val, y_nonseq_val_pred)
    valid_acc = np.mean(y_nonseq_val == y_nonseq_val_pred)
    valid_f1 = f1_score(y_nonseq_val, y_nonseq_val_pred, average="macro")
                
    pretrain_epochs = 0
    epoch = 10
    # Report performance
    print_performance(n_train_examples, n_valid_examples, train_cm, 
                      valid_cm, epoch, pretrain_epochs, train_duration, 
                      train_loss, train_acc, train_f1, valid_duration, 
                      valid_loss, valid_acc, valid_f1)    
        
    total_cm += train_cm
    # val_cm += valid_cm

    print(total_cm)

time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum)

