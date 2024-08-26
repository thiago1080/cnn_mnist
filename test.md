- conv1: convolutional layer with input shape (28, 28, 32)

- conv_2_1:  convolutional layer with input shape (14, 14, 32)
- conv_2_2:  a convolutional layer with input shape (14, 14, 32)

- conv_3_1:  convolutional layer with input shape (7, 7, 256)
- conv_3_2:  a convolutional layer with input shape (7, 7, 256)
- conv_3:  a convolutional layer which is the concatenation between conv_3_1 and conv_3_2 with shape (7, 7, 256)

- fc_1: fully connected layer with shape (1000, )
- fc_2: fully connected layer with shape (500, )
- output: output layer with shape (10, )


connections: 
    conv1 -> conv_2_1
    conv1 -> conv_2_2
    conv2 -> conv_3_1
    conv2 -> conv_3_2
    concatenate(conv_3_1, conv_3_2) -> fc_1
    fc_1 -> fc_2
    fc_2 -> output