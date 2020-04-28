# Image Caption
This part of the project contains a 'caption generator'. Its architecture is as follows.

<img src="https://drive.google.com/uc?id=1ja11_VcP4xvSwZGFfLkUnwM6Mi48Qyu5" data-canonical-src="https://drive.google.com/uc?id=1ja11_VcP4xvSwZGFfLkUnwM6Mi48Qyu5" width="400" height="400" />

The underlying VGG16 network trained on ImageNet, is used via 'transfer learning' to help with object classification. The final layer has been replaced, and its features are directly fed into our network. This net can be one of VGG16 or InceptionV3.<br/>
The rest of of the model is custom.

### Input Layers
There are two input layers as follows.
* Input Layer 1: Image to be captioned. This is fed into the underlying InceptionV3 net.
* Input Layer 2: Caption generated thus far.

### Key Hidden Layers
The output of the InceptionV3 and LSTM nets are combined and fed into dense layer witu ReLU activation.

### Output Layer
The output of the net is a Softmax over all words in the vocabulary.

### Goodness of Fit
Goodness of fit is measured via the BLEU score.
<br/><br/>

# References

Papineni K., Roukos S., Ward T., Zhu W. (Jul 2002), [BLEU: a method for automatic evaluation of machine translation, Proceedings of the 40th Annual Meeting on Association for Computational Linguistics](https://dl.acm.org/doi/10.3115/1073083.1073135)

Szegedy C et al., (Sep 2014), [Going Deeper with Convolutions,  arXiv:1409.4842 cs.CV](https://arxiv.org/abs/1409.4842)

Simonyan K, Zisserman A (Sep 2014), [Very Deep Convolutional Networks for Large-Scale Image Recognition, arXiv:1409.1556 cs.CV](https://arxiv.org/abs/1409.1556)

Tanti M, Gatt A, Camilleri K.P., (Aug 2017), [What is the Role of Recurrent Neural Networks (RNNs) in an Image Caption Generator?, arXiv:1708.02043 cs.CL](https://arxiv.org/abs/1708.02043)

Brownlee J. (Jun 2019), [How to Develop a Deep Learning Photo Caption Generator from Scratch, Machine Learning Mastery](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)

Plummer B. et al., (May 2015), [Flickr30K Entities: Collecting Region-to-Phrase Correspondences for Richer Image-to-Sentence Models, arXiv:1505.04870 cs.CV](https://arxiv.org/abs/1505.04870)
