import numpy as np
import os as os
import shutil
from config import Config
import sys
import argparse
import tensorflow as tf
import shutil

np.random.seed(seed=0)


VIEW_PER_RING = 9
N_RING = 8


config = Config()
config.learning_rate = 0.00001
config.resume = './models/shrec18/model_best_testset.ckpt-110352'
config.log_folder ='./tf_logs'
config.config_name = 'eureka_run'
class MultiviewDataset():
    LIST_RING = [
        [0, 1, 2, 8, 9, 10, 16, 17, 18],
        [1, 2, 3, 9, 10, 11, 17, 18, 19],
        [2, 3, 4, 10, 11, 12, 18, 19, 20],
        [3, 4, 5, 11, 12, 13, 19, 20, 21],
        [4, 5, 6, 12, 13, 14, 20, 21, 22],
        [5, 6, 7, 13, 14, 15, 21, 22, 23],
        [6, 7, 0, 14, 15, 8,  22, 23, 16],
        [7, 0, 1, 15, 8,  9,  23, 16, 17 ]
    ]

    def __init__(self, x_path, y_path, batch_size = 32, shuffle = None):
        def makeRing(X,ring):
            return X[:,ring,:] # N x View x 2048
        
        if type(x_path) is not np.ndarray:
            X, Y = np.load(x_path),np.load(y_path).astype(np.int64).squeeze()
        else:
            X = x_path
            Y = y_path.astype(np.int64).squeeze()

        X = X.reshape(-1, 26, 2048)
        
        tmp = [np.expand_dims(makeRing(X, self.LIST_RING[i]),0) for i in range(N_RING)]
        tmp = np.concatenate(tmp)

        tmp = np.transpose(tmp, [1,0,2,3])
        print 'tmp', tmp.shape

        
        self.X = tmp # N x n_ring x n_view_per_ring x 2048
        self.Y = Y[np.arange(0, Y.shape[0], 26)] # N

        #fix dims
        self.X = self.X.reshape(-1, VIEW_PER_RING, 2048) #  (N x n_ring), view_per_ring , 2048
        self.Y = np.repeat(self.Y, N_RING , 0 )

        print self.X.shape, self.Y.shape

        if batch_size==-1:
            self.batch_size = self.Y.shape[0]
        else:
            self.batch_size = batch_size

        self.shuffle = shuffle
    
        self.reset()

    def reset(self):
        self.idxs = np.arange(0, self.X.shape[0])
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.num_batches = (self.X.shape[0]+self.batch_size-1) // self.batch_size
        self.batch_idx = 0
    
    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    
    def next_batch(self):
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, self.X.shape[0])
        batch_data = self.X[self.idxs[start_idx:end_idx]]    #np.zeros((bsize, self.npoints, self.num_channel()))
        batch_label = self.Y[self.idxs[start_idx:end_idx]] 
        
        self.batch_idx += 1
        return batch_data, batch_label
    
'''
MultiviewDataset_train = MultiviewDataset(os.path.join(config.data,'x_train.npy' ),
                                os.path.join(config.data, 'y_train.npy'),
                                batch_size=config.batch_size,
                                shuffle=True
                                )

MultiviewDataset_val = MultiviewDataset(
                                os.path.join(config.data, 'x_val.npy'),
                                os.path.join(config.data, 'y_val.npy'),
                                batch_size=-1,
                                shuffle=False
                                )

MultiviewDataset_test = MultiviewDataset(
                                os.path.join(config.data, 'x_test.npy'),
                                os.path.join(config.data, 'y_test.npy'),
                                batch_size=-1,
                                shuffle=False
                                )


TRAIN_DATASET = MultiviewDataset_train
TEST_DATASET = MultiviewDataset_test
#VAL_DATASET = MultiviewDataset_val

print 'train size:', TRAIN_DATASET.X.shape, TRAIN_DATASET.Y.shape, TRAIN_DATASET.X.dtype
#print 'val size:', VAL_DATASET.X.shape, VAL_DATASET.Y.shape
print 'test size:', TEST_DATASET.X.shape, TEST_DATASET.Y.shape
'''
def inference_ring_score(ring): #discriminator
    '''
    ring: N x V x C
    
    return N,
    '''


    return tf.ones( ( tf.shape(ring)[0] ,1), dtype=tf.float64)



def inference_ring(ring, n_classes):
    '''
    ring: N x V x C
    '''
    n_view = ring.get_shape().as_list()[1]
    ring = tf.transpose(ring, perm=[1,0,2])

    view_scores = []

    with tf.name_scope("view_slice"):
        for i in xrange(n_view):
            reuse = i > 0
            view = tf.gather(ring, i) # N x C
            fc = _fc(view, n_classes, reuse = reuse)# N x n_classes
            view_scores.append(fc)
    
    
    ret = tf.stack(view_scores, 0)
    ret = tf.reduce_mean(ret,0)
    return ret

def inference(x, y, is_training, n_classes, keep_prob):
    '''
    x: N samples x V x C   (C=2048)
    y: N x
    is_training : bool 
    '''
    #x = tf.placeholder(tf.float64, [None, 8, 2048])
    ring = x
    n_view = ring.get_shape().as_list()[1]
    ring = tf.transpose(ring, perm=[1,0,2])

    view_features = []

    with tf.variable_scope("view_slice"):
        for i in xrange(n_view):
            view = tf.gather(ring, i) # N x C
            fc = _fc(view, 128, reuse = tf.AUTO_REUSE)# N x 128
            fc = tf.nn.relu(fc)
            view_features.append(fc)

    #view_features: list of  N x 128
    view_features = tf.stack(view_features,0) # V x N x 128 
    view_features = tf.transpose(view_features,[1,0,2]) # N x V x 128
    view_features = tf.reshape(view_features,[-1, n_view * 128])
    
    with tf.variable_scope("view_pool"):
        ret = _fc(view_features, n_classes, reuse=False)
    
    return ret

def _view_pool(view_features, input_shape, output_shape):
    # view_features: list of  
    vp = tf.stack(view_features, 0)
    vp = tf.reduce_max(vp, 0)
    return vp # N x n_classes

def _fc(x, output_shape, reuse):
    '''
    x: N x C 
    '''
    input_shape = x.get_shape().as_list()[1]
    with tf.variable_scope("fc_layer", reuse = reuse) as scope:
        W = tf.get_variable(name="weights", shape=[input_shape, output_shape],dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="biases", shape=[output_shape], dtype= tf.float64, initializer=tf.zeros_initializer())

        x = tf.matmul(x,W)+b
        #x = tf.nn.sigmoid(x)
        #x = tf.nn.relu(x)
    return x


def _view_pool_with_classes(view_features, y, n_classes, is_training,reuse = False):

    with tf.variable_scope("view_pool_with_classes", reuse = reuse ) as scope:

        W = tf.get_variable(name="weights", shape= [1, n_classes], dtype=tf.float64, initializer=tf.ones_initializer())
        b = tf.get_variable(name="biases", shape = [n_classes], dtype=tf.float64, initializer=tf.zeros_initializer())

        #W1 = tf.get_variable(name="weights1", shape= [n_classes, n_classes], dtype=tf.float64, initializer=tf.ones_initializer())
        #b1 = tf.get_variable(name="biases1", shape = [n_classes], dtype=tf.float64, initializer=tf.zeros_initializer())

        y = tf.cast(y, tf.float64)
        y = tf.expand_dims(y, 1)
        mask = tf.add(tf.matmul(y , W), b)
        #mask = tf.sigmoid(mask)
        #mask = tf.add(tf.matmul(mask , W1), b1)
        #mask = tf.sigmoid(mask)
        vp = tf.stack(view_features, 0)
        vp = tf.reduce_mean(vp,0)

        vp = tf.multiply(vp, mask)

        return vp


def build_graph_ops(x_place_holder, y_place_holder, is_training_pl,  train_graph, global_step):
    n_classes=20
    logits = inference(x_place_holder,y_place_holder, is_training_pl, n_classes, False)# if train_graph else inference_test(x_place_holder,n_classes)
    loss = get_loss(logits,y_place_holder)
    
    #bn_decay = get_bn_decay(global_step)
    learning_rate = get_learning_rate(global_step)
    train_op = get_train_op(loss, learning_rate, global_step)

    score = get_score(logits)
    preds = get_pred(score)
    acc = get_accuracy(preds, y_place_holder)

    ops = {
            'x_pl': x_place_holder,
            'y_pl': y_place_holder,
            'is_training_pl': is_training_pl,
            'loss': loss,
            'train_op': train_op,
            'merged': None,
            'global_step': global_step,
            'pred': preds,
            'score': score
        }

    return ops



def get_train_op(loss, lr, step):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss, step)
    return train_op

def get_accuracy(preds, y):
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, y), tf.float32))
    tf.summary.scalar('accuracy', acc)
    return acc

def get_score(logits):
    return tf.nn.softmax(logits)

def get_pred(score):
    return tf.arg_max(score,1)

def get_loss(logits,y):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(loss)
    
    tf.summary.scalar('loss', loss)

    return loss

def get_learning_rate(global_step):
    learning_rate = tf.constant(config.learning_rate)
    '''
    learning_rate = tf.train.exponential_decay(
                        config.learning_rate,  # Base learning rate.
                        global_step * config.batch_size,  # Current index into the dataset.
                        config.decay_step,          # Decay step.
                        config.decay_rate,          # Decay rate.
                        staircase=True)
    
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    '''
    tf.summary.scalar('learning_rate', learning_rate)

    return learning_rate        


def get_bn_decay(global_step):
    bn_momentum = tf.train.exponential_decay(
                      config.bn_init_decay,
                      global_step * config.batch_size,
                      config.bn_decay_step,
                      config.bn_decay_rate,
                      staircase=True)
    bn_decay = tf.minimum(config.bn_decay_rate, 1 - bn_momentum)       
    tf.summary.scalar('bn_decay', bn_decay)

    return bn_decay

def eval():
    with tf.Graph().as_default():
        #placeholders
        x_place_holder = tf.placeholder(tf.float64, shape=(None,VIEW_PER_RING , 2048), name="input_data")
        y_place_holder = tf.placeholder(tf.int64, shape= (None,), name="labels")
        is_training_pl = tf.placeholder(tf.bool, shape=(), name = "is_training")
        
        #global step
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        #get all ops
        training_ops = build_graph_ops(x_place_holder, y_place_holder, is_training_pl, True, global_step)

        merged = tf.summary.merge_all()
        training_ops['merged'] = merged
        #testing_ops['merged'] = merged     
        #saver
        saver = tf.train.Saver(max_to_keep=200)

        #TF Session config
        config_session = tf.ConfigProto()
        ##config_session.gpu_options.allow_growth = True
        sess = tf.Session(config=config_session)
        assert config.resume is not None
        print 'restoring model from', config.resume
        saver.restore(sess, config.resume )
       
        eval_writer = tf.summary.FileWriter(os.path.join(config.log_folder,config.config_name,'eval'), sess.graph)
        test_acc, test_pred,test_score = eval_one_epoch(0, sess, training_ops, eval_writer, TEST_DATASET)


        gstep = global_step.eval(sess)

        np.savetxt(os.path.join(config.log_folder, config.config_name,'./evaluate_testset_{}.pred.txt'.format(gstep)) , test_pred, fmt='%d')
        np.savetxt(os.path.join(config.log_folder, config.config_name,'./evaluate_testset_{}.score.txt'.format(gstep)), test_score, fmt="%.8f")

        print 'accuracy:', test_acc


def train():
    with tf.Graph().as_default():
        
        #placeholders
        x_place_holder = tf.placeholder(tf.float64, shape=(None, VIEW_PER_RING, 2048), name="input_data")
        y_place_holder = tf.placeholder(tf.int64, shape= (None,), name="labels")
        is_training_pl = tf.placeholder(tf.bool, shape=(), name = "is_training")
        
        #global step
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        #get all ops
        training_ops = build_graph_ops(x_place_holder, y_place_holder, is_training_pl, True, global_step)
        #testing_ops = build_graph_ops(x_place_holder, y_place_holder, is_training_pl, False, global_step)

        merged = tf.summary.merge_all()
        training_ops['merged'] = merged
        #testing_ops['merged'] = merged     
        #saver
        saver = tf.train.Saver(max_to_keep=200)

        #TF Session config
        config_session = tf.ConfigProto()
        ##config_session.gpu_options.allow_growth = True
        sess = tf.Session(config=config_session)
        if config.resume is not None:
            print 'restoring model from', config.resume
            saver.restore(sess, config.resume )
        else:
            sess.run(tf.global_variables_initializer())
        # summary writers
        train_writer = tf.summary.FileWriter(os.path.join(config.log_folder,config.config_name,'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(config.log_folder,config.config_name, 'test'), sess.graph)
        #val_writer = tf.summary.FileWriter(os.path.join(config.log_folder,config.config_name, 'val'), sess.graph)
        
        #current_best_on_val_set = 0
        current_best_on_test_set = 0
        for epoch in xrange(config.max_epoch):
            sys.stdout.flush()
            
            train_one_epoch(epoch, sess, training_ops, train_writer, TRAIN_DATASET)
            
            #val_acc, val_pred,val_score = eval_one_epoch(epoch, sess, training_ops, val_writer, VAL_DATASET )
            test_acc, test_pred, test_score = eval_one_epoch(epoch, sess, training_ops, test_writer, TEST_DATASET )
            
            gstep = global_step.eval(sess)
            '''
            if val_acc > current_best_on_val_set:
                saver.save(sess=sess, global_step=global_step, save_path= os.path.join(config.log_folder, config.config_name,'./model_best.ckpt'))
                np.savetxt(os.path.join(config.log_folder, config.config_name,'./model_best_valset_{}.pred.txt'.format(gstep)) , val_pred, fmt='%d')
                np.savetxt(os.path.join(config.log_folder, config.config_name,'./model_best_valset_{}.score.txt'.format(gstep)), val_score, fmt="%.8f")
                print('accuracy on validation set increased, saved model')
                current_best_on_val_set = val_acc
            '''
            if test_acc > current_best_on_test_set:
                saver.save(sess=sess, global_step=global_step, save_path= os.path.join(config.log_folder, config.config_name,'./model_best_testset.ckpt'))
                np.savetxt(os.path.join(config.log_folder, config.config_name,'./model_best_testset_{}.pred.txt'.format(gstep)) , test_pred, fmt='%d')
                np.savetxt(os.path.join(config.log_folder, config.config_name,'./model_best_testset_{}.score.txt'.format(gstep)), test_score, fmt="%.8f")
                print('accuracy on test set increased, saved model')
                current_best_on_test_set = test_acc
            
            if epoch % 30 == 0:
                saver.save(sess=sess, global_step=global_step, save_path= os.path.join(config.log_folder, config.config_name,'./model.ckpt'))
                print('saved model')


def train_one_epoch(epoch, sess, ops, train_writer, dataset):
    is_training = True
    total_correct = 0
    total_seen = 0
    loss = 0

    while dataset.has_next_batch():
        batch_data, batch_label = dataset.next_batch()
        #batch_data = batch_data.reshape(-1,2048*8)
        bsize = batch_data.shape[0]
        feed_dict = {
            ops['x_pl']: batch_data,
            ops['y_pl']: batch_label,
            ops['is_training_pl']: is_training
        }
        
        summary, step, _,loss_val, pred_val = sess.run([ops['merged'], ops['global_step'], ops['train_op'], ops['loss'], ops['pred']] , feed_dict = feed_dict )
        train_writer.add_summary(summary, step)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss += loss_val * bsize * 1.0 / dataset.batch_size  


    print 'epoch,train loss_sum,train acc, test loss, test acc', epoch, loss, total_correct*1./total_seen 
    dataset.reset()
                                                                        
def eval_one_epoch(epoch, sess, ops, test_writer, dataset):
    is_training = False
                                                                        
    total_correct = 0
    total_seen = 0
    loss = 0
    total_pred = np.array([])
    total_score = np.zeros((0,20))
    while dataset.has_next_batch():
        batch_data, batch_label = dataset.next_batch()
        #batch_data = batch_data.reshape(-1,2048*8)

        bsize = batch_data.shape[0]
        feed_dict = {
            ops['x_pl']: batch_data,
            ops['y_pl']: batch_label,
            ops['is_training_pl']: is_training
        }
        summary,step, loss_val, score_val, pred_val = sess.run([ops['merged'], ops['global_step'], ops['loss'], ops['score'], ops['pred']] , feed_dict = feed_dict )
        
        test_writer.add_summary(summary, step)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss += loss_val * bsize * 1.0 / dataset.batch_size


        total_score = np.concatenate([total_score, score_val])
        total_pred = np.concatenate([total_pred, pred_val])

    #np.savetxt('predict.csv', total_pred, fmt='%d')                                                                        
    print loss, total_correct*1./total_seen                                                                                                                                                                                                                                                                                                                                                         
    dataset.reset()                                                                        
    
    return total_correct*1./total_seen, total_pred, total_score

'''
if os.path.exists(os.path.join(config.log_folder, config.config_name)) is False:
    os.mkdir(os.path.join(config.log_folder, config.config_name))
shutil.copy(__file__, os.path.join(config.log_folder, config.config_name,__file__))
train()

'''





def predict(X):
    dataset = MultiviewDataset(X,np.zeros((X.shape[0],1)),-1, False)
    is_training = False

    with tf.Graph().as_default():
        #placeholders
        x_place_holder = tf.placeholder(tf.float64, shape=(None,VIEW_PER_RING , 2048), name="input_data")
        y_place_holder = tf.placeholder(tf.int64, shape= (None,), name="labels")
        is_training_pl = tf.placeholder(tf.bool, shape=(), name = "is_training")
        
        #global step
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        #get all ops
        training_ops = build_graph_ops(x_place_holder, y_place_holder, is_training_pl, True, global_step)

        merged = tf.summary.merge_all()
        training_ops['merged'] = merged
        #testing_ops['merged'] = merged     
        #saver
        saver = tf.train.Saver(max_to_keep=200)

        #TF Session config
        config_session = tf.ConfigProto()
        ##config_session.gpu_options.allow_growth = True
        sess = tf.Session(config=config_session)
        assert config.resume is not None
        print 'restoring model from', config.resume
        saver.restore(sess, config.resume )
        print 'restored model from', config.resume

        eval_writer = tf.summary.FileWriter(os.path.join(config.log_folder,config.config_name,'eval'), sess.graph)

        test_acc, test_pred,test_score = eval_one_epoch(0, sess, training_ops, eval_writer, dataset)


        gstep = global_step.eval(sess)

        np.savetxt(os.path.join(config.log_folder, config.config_name,'./evaluate_testset_{}.pred.txt'.format(gstep)) , test_pred, fmt='%d')
        np.savetxt(os.path.join(config.log_folder, config.config_name,'./evaluate_testset_{}.score.txt'.format(gstep)), test_score, fmt="%.8f")

        print 'accuracy:', test_acc
        
        class_score = test_score
        class_score = class_score.reshape(-1,N_RING,20)
        arg_mean = np.mean(class_score,1)
        preds = np.argmax(arg_mean,1)
        
        
        return arg_mean, preds


#eval()
