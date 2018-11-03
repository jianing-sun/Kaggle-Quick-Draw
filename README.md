# Quick Draw Kaggle Competition 

## TODO

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

half data augmentation, 2/3 data augmentation were added and trained. The latter one I used with SGD and the former one used Adam. Got improved from 0.916 to 0.923 after training 38500 iterations with batch_size=1024, 2/3 data augmentation, SGD. 

- [x] noise removal based on image entropy

done

- [ ] generate last layer features vector for each class

ummm, I wrote the code to extract last layer features but for each category is cost half an hour if I calculate with full csv file. I won't finish for all classes even after the ddl of the competition. Should I reduce the number of instances used? like instead reading each full csv, try to use with limited rows? or just give up.

- [x] generate sklearn classification report 

done. it has been save into a .csv file included under the project folder. Some of the classes have pretty low precision, e.g. bus, the great wall of China. Should I use weighted loss function to compenstate the imbalance of instsance number of each class?

- [ ] ~~Loss Functions for Top-k Error: Analysis and Insights~~

http://openaccess.thecvf.com/content_cvpr_2016/papers/Lapin_Loss_Functions_for_CVPR_2016_paper.pdf
didnt understand: it is a generalization from categorical cross-entropy loss? I feel it's not. it's more inclined to OVA (one-versus-all) multiclassification method, got confused by the definition of 'classifer'


- [ ] Implement Shakedrop Regularization on Resnet50

shake-shake regularization: https://arxiv.org/pdf/1705.07485.pdf
shake-drop regularization: https://arxiv.org/pdf/1802.02375.pdf






- new/interesting ideas
  https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/68006
