## Quick Draw Kaggle Competition 

- TODO: 
- [x] ~~generate sklearn report to evaluate valid_set and find the problem of overfitting or others~~ data augmentation, flip, rotate, stroke removal with prob, image global deformation

note: there are some good data exploration kernel in Kaggle, e.g. https://www.kaggle.com/chenling0927/what-drawings-are-hard-to-predict, so it's unecessary to do sth similar to what they've done. Basiclly the reason of wrong prediction is some of them are random doodle, many have words in it, and some of them are too complicated with so many storkes. This inspires me that the performance is likely to be improved with appropriate data preprocessing, e.g. data augmentation, stroke removal and image global deformation. 



- [x] ~~DRNN (Dilated Residual NN) / NASNet~~ ResNet 50 layers, should I change the input dimension? 

note: NASNet is so big with more than 83 million parameters and I am not sure whether DRNN would has a large improvement. So currently I won't change the network model from RNN50 to avoid wasting time on training with different but uncertain models. (I prefer to use the one I'm familiar with. =) 

- [x] ~~Change input and output layer with countrycode feature input and predicition~~

note: after browsing several paper about sketch & free hand-writing recognition, none of them use the country code feature, so for now just put a pin in that. 

- [x] ~~custom loss with country code~~

ditto

- [ ] custom loss with sketch center loss 

note: Inspired from https://link.springer.com/content/pdf/10.1007%2Fs11263-016-0932-3.pdf and http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2763.pdf 

- [x] data augmentation

random data augmentation among null, random rotate, random flip and global deformation

- new/interesting ideas
  https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/68006
