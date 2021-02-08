import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow import feature_column
from tensorflow.keras import regularizers

num_bin_size = (64, 16, 128, 64, 128, 64, 512, 512)
cate_bin_size = (512, 128, 256, 256, 64, 256, 256, 16, 256)


class MLP(Model):

    def __init__(self, name, params):
        super(MLP, self).__init__()
        self.model_name = name
        self.params = params
        num_features = [feature_column.bucketized_column(
            feature_column.numeric_column(str(i)),
            boundaries=[j*1.0/(num_bin_size[i]-1) for j in range(num_bin_size[i]-1)])
            for i in range(8)]
        if name == "MLP_FSIW":
            print("using elapse feature")
            num_features.append(feature_column.numeric_column("elapse"))
        cate_features = [feature_column.embedding_column(
            feature_column.categorical_column_with_hash_bucket(
                str(i), hash_bucket_size=cate_bin_size[i-8]),
            dimension=8) for i in range(8, 17)]

        all_features = num_features + cate_features

        self.feature_layer = tf.keras.layers.DenseFeatures(all_features)

        self.fc1 = layers.Dense(256, activation=tf.nn.leaky_relu,
                                kernel_regularizer=regularizers.l2(params["l2_reg"]))
        self.bn1 = layers.BatchNormalization()
        self.fc2 = layers.Dense(256, activation=tf.nn.leaky_relu,
                                kernel_regularizer=regularizers.l2(params["l2_reg"]))

        if self.model_name == "MLP_wintime_sep":
            self.bn21 = layers.BatchNormalization()
            self.fc31 = layers.Dense(128, activation=tf.nn.leaky_relu,
                                kernel_regularizer=regularizers.l2(params["l2_reg"]))
            self.bn31 = layers.BatchNormalization()
            self.bn22 = layers.BatchNormalization()
            self.fc32 = layers.Dense(128, activation=tf.nn.leaky_relu,
                                kernel_regularizer=regularizers.l2(params["l2_reg"]))
            self.bn32 = layers.BatchNormalization()
        #elif self.model_name == "MLP_wintime_sep":
            
        else:
            self.bn2 = layers.BatchNormalization()
            self.fc3 = layers.Dense(128, activation=tf.nn.leaky_relu,
                                kernel_regularizer=regularizers.l2(params["l2_reg"]))
            self.bn3 = layers.BatchNormalization()

        print("build model {}".format(name))
        if self.model_name == "MLP_EXP_DELAY":
            self.fc4 = layers.Dense(2)
        elif self.model_name == "MLP_tn_dp":
            self.fc4 = layers.Dense(2)
        elif self.model_name == "MLP_3class":
            self.fc4 = layers.Dense(3)
        elif self.model_name in ["MLP_SIG", "MLP_FSIW"]:
            self.fc4 = layers.Dense(1)
        elif self.model_name == "MLP_likeli":
            self.fc4 = layers.Dense(2)
        elif self.model_name == "MLP_wintime_sep":
            self.fc41 = layers.Dense(1)
            self.fc42 = layers.Dense(1)
        elif self.model_name == "MLP_winadapt":
            self.fc4 = layers.Dense(4)
            #print 'MLP_winadapt before out'
            #print self.fc4
            
        else:
            raise ValueError("model name {} not exist".format(name))

    def call(self, x, training=True):
        x = self.feature_layer(x)
        x = self.fc1(x)
        x = self.bn1(x, training=training)
        x = self.fc2(x)
        if self.model_name == "MLP_wintime_sep":
            x1 = self.bn21(x, training=training)
            x1 = self.fc31(x1)
            x1 = self.bn31(x1, training=training)
            x1 = self.fc41(x1)

            x2 = self.bn22(x, training=training)
            x2 = self.fc32(x2)
            x2 = self.bn32(x2, training=training)
            x2 = self.fc42(x2)

            x = tf.concat([x1,x2], axis=1)
        else:
            x = self.bn2(x, training=training)
            x = self.fc3(x)
            x = self.bn3(x, training=training)
            x = self.fc4(x)
        if self.model_name == "MLP_EXP_DELAY":
            return {"logits": tf.reshape(x[:, 0], (-1, 1)), "log_lamb": tf.reshape(x[:, 1], (-1, 1))}
        elif self.model_name in ["MLP_SIG", "MLP_FSIW", "MLP_3class","MLP_winadapt"]:
            #print 'MLP_winadapt out'
            #print x.shape
            #print x
            return {"logits": x}
        elif self.model_name == "MLP_tn_dp":
            return {"tn_logits": tf.reshape(x[:, 0], (-1, 1)), "dp_logits": tf.reshape(x[:, 1], (-1, 1))}
        elif self.model_name == "MLP_likeli":
            return {"logits": x, "cv_logits": tf.reshape(x[:, 0], (-1, 1)), "time_logits": tf.reshape(x[:, 1], (-1, 1))}
        elif self.model_name == "MLP_wintime_sep":
            return {"logits": x, "cv_logits": x1, "time_logits": x2}
        else:
            raise NotImplementedError()

    def predict(self, x):
        return self.call(x, training=False)["logits"]



def get_model(name, params):
    if name in ["MLP_EXP_DELAY", "MLP_SIG", "MLP_tn_dp", "MLP_FSIW", "MLP_3class", "MLP_likeli", "MLP_wintime_sep", "MLP_winadapt"]:
        return MLP(name, params)
    else:
        raise NotImplementedError()
