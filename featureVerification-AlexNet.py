#some basic imports and setups
# import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataRead import *
import json
from datetime import datetime



# def featureVerification(model,face1S,face2S,fileName):
#     with tf.Session() as sess:
        
#         # Initialize all variables
#         sess.run(tf.global_variables_initializer())
        
#         # Load the pretrained weights into the model
#         model.load_initial_weights(sess)
        
#         # Loop over all images
#         for i in range(2):

#             feaArray1[i]=sess.run(model.fc7, feed_dict={x: [face1S[i]], keep_prob: 1})[0]
#             # print(np.size(feaArray1[i]),feaArray1[i])
#             feaArray2[i]=sess.run(model.fc7, feed_dict={x: [face2S[i]], keep_prob: 1})[0]
#     i=2
#     # print(np.size(feaArray1),feaArray1)
#     y_test,y_score=feaVeri(feaArray1[:i],feaArray2[:i])
#     # print(y_test,y_score)
#     d={}
#     d['y_test']=np.array2string(y_test)
#     d['y_score']=np.array2string(y_score)
#     with open(fileName,'w')as f:
#         json.dump(d,f)
#     # f,t,_=roc_curve(y_test, y_score)
#     # ROCfigure(f,t)
# # def fineTune(model,trainingX,trainingY):
if __name__ == '__main__':

    face1S_Ori,face2S_Ori,lableS=read2()
    # print(lableS)
    size4eachClass=64


    # Learning params
    learning_rate = 0.1
    num_epochs = 100
    batch_size = size4eachClass*2

    # Network params
    dropout_rate = 0.5
    num_classes = 2
    train_layers = ['fc8', 'fc7']




    face1S=resizeBy0(face1S_Ori,32,32,227,227)[:size4eachClass]
    face2S=resizeBy0(face2S_Ori,32,32,227,227)[:size4eachClass]


    feaArray=[0]*size4eachClass*2
    feaArray1=[0]*size4eachClass
    feaArray2=[0]*size4eachClass
    # print(face1S.shape)
    trainingX_orig=np.append(face1S,face2S,axis=0)
    trainingX=[]
    trainingY_orig=[[1,0] for i in range(size4eachClass)]+[[0,1] for i in range(size4eachClass)]
    trainingY=[]
    f1=[]
    t1=[]
    # print(trainingX.shape)
    
        
    from alexnet import AlexNet

    #placeholder for input and dropout rate
    x = tf.placeholder(tf.float32, [size4eachClass*2, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)

    #create model with default config ( == no skip_layer and 1000 units in the last layer)
    model = AlexNet(x, keep_prob, 2, ['fc8'])
    # featureVerification(model,face1S,face2S,'ROC_Original.json')



    # # --------------------------------------inital feature verification-----------------------------------------
    

    # with tf.Session() as sess:
        
        # Initialize all variables
        # sess.run(tf.global_variables_initializer())
        
        # # Load the pretrained weights into the model
        # model.load_initial_weights(sess)
        
    #     feaArray=sess.run(model.fc7, feed_dict={x: trainingX_orig, keep_prob: 1})
    #     # print(feaArray.shape)
    #         # print(np.size(feaArray1[i]),feaArray1[i])
    #         # feaArray2[i]=sess.run(model.fc7, feed_dict={x: [face2S[i]], keep_prob: 1})[0]
    # # print(np.size(feaArray1),feaArray1)
    # y_test,y_score=feaVeri(feaArray[:size4eachClass],feaArray[size4eachClass:])
    # # print(y_test,y_score)
    # d={}
    # d['y_test']=np.array2string(y_test)
    # d['y_score']=np.array2string(y_score)
    # with open('ROC_Original.json','w')as f:
    #     json.dump(d,f)
    # f1,t1,_=roc_curve(y_test, y_score)
    # ----------------------------------------------fineTune----------------------------------------
    
    # shuffle
    indexRand = np.random.permutation(size4eachClass*2)
    # print(indexRand)
    for i in indexRand:
        trainingX.append(trainingX_orig[i])
        trainingY.append(trainingY_orig[i])
    # print(trainingY)


    # How often we want to write the tf.summary data to disk
    display_step = 1
    filewriter_path = "./finetune_alexnet/"
    checkpoint_path = "./finetune_alexnet/"

    # TF placeholder for graph input and output
    # x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [None, num_classes])
    # keep_prob = tf.placeholder(tf.float32)

    # Initialize model
    # model = AlexNet(x, keep_prob, num_classes, train_layers)

    # Link variable to model output
    score = model.fc8

    # List of trainable variables of the layers we want to train
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

    # Op for calculating the loss
    with tf.name_scope("cross_ent"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))  

    # Train op
    with tf.name_scope("train"):
        # Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))

        # Create optimizer and apply gradient descent to the trainable variables
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    # Add gradients to summary  
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradient', gradient)

    # Add the variables we train to the summary  
    for var in var_list:
        tf.summary.histogram(var.name, var)
      
    # Add the loss to summary
    tf.summary.scalar('cross_entropy', loss)
    

    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
      
    # Add the accuracy to the summary
    tf.summary.scalar('accuracy', accuracy)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Initalize the data generator seperately for the training and validation set
    # train_generator = ImageDataGenerator(train_file, 
                                         # horizontal_flip = True, shuffle = True)
    # val_generator = ImageDataGenerator(val_file, shuffle = False) 

    # Get the number of training/validation steps per epoch
    # train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
    # val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)
    train_batches_per_epoch =1
    val_batches_per_epoch =1

    # Start Tensorflow session
    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        # Load the pretrained weights into the non-trainable layer
        model.load_initial_weights(sess)


        feaArray=sess.run(model.fc7, feed_dict={x: trainingX_orig, keep_prob: 1})
            # print(feaArray.shape)
                # print(np.size(feaArray1[i]),feaArray1[i])
                # feaArray2[i]=sess.run(model.fc7, feed_dict={x: [face2S[i]], keep_prob: 1})[0]
        # print(np.size(feaArray1),feaArray1)
        y_test,y_score=feaVeri(feaArray[:size4eachClass],feaArray[size4eachClass:])
        # print(y_test,y_score)
        d={}
        d['y_test']=np.array2string(y_test)
        d['y_score']=np.array2string(y_score)
        with open('ROC_Original.json','w')as f:
            json.dump(d,f)
        f1,t1,_=roc_curve(y_test, y_score)


        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(), 
                                                        filewriter_path))

        # Loop over number of epochs
        for epoch in range(num_epochs):

            print("{} Epoch number: {}".format(datetime.now(), epoch+1))
            
            step = 0
            
            while step < train_batches_per_epoch:
                
                # Get a batch of images and labels
                # batch_xs, batch_ys = train_generator.next_batch(batch_size)
                
                # And run the training op
                sess.run(train_op, feed_dict={x: trainingX, 
                                              y: trainingY, 
                                              keep_prob: dropout_rate})
                # print('??',lossVal)
                # Generate summary with the current batch of data and write to file
                if step%display_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: trainingX, 
                                                            y: trainingY, 
                                                            keep_prob: 1.})
                    writer.add_summary(s, epoch*train_batches_per_epoch + step)
                    
                step += 1
            
            # # --------------------------------------feature verification-----------------------------------------
        feaArray=sess.run(model.fc7, feed_dict={x: trainingX_orig, keep_prob: 1})
        # print(feaArray.shape)
            # print(np.size(feaArray1[i]),feaArray1[i])
            # feaArray2[i]=sess.run(model.fc7, feed_dict={x: [face2S[i]], keep_prob: 1})[0]
    # print(np.size(feaArray1),feaArray1)
    y_test,y_score=feaVeri(feaArray[:size4eachClass],feaArray[size4eachClass:])
    # print(y_test,y_score)
    d={}
    d['y_test']=np.array2string(y_test)
    d['y_score']=np.array2string(y_score)
    with open('ROC_tuned.json','w')as f:
        json.dump(d,f)
    f2,t2,_=roc_curve(y_test, y_score)
    ROCfigure(f1,t1,f2,t2)
