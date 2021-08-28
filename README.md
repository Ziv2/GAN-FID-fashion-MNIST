# GAN for fashion-MNIST, with FID
Attached is a basic GAN, referenced to "MNIST FASION".

Since the discriminator also improves during training, I needed an *external tool* to evaluate the improvement of the genreted images.

To do this I used FID.

https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance

Following are fewpictures to illustrate the training progress.

Epoch#0:
![Epoch#0](https://user-images.githubusercontent.com/41025885/131232494-e1ad44c1-a9c8-419c-9aa2-e6e07921ece8.png)
Epoch#3:
![Epoch#3](https://user-images.githubusercontent.com/41025885/131232496-898ba6b7-26a4-4011-bd9d-5edf40ed3ed7.png)
Epoch#10:
![Epoch#10](https://user-images.githubusercontent.com/41025885/131232519-c92ad681-893d-4817-aa2e-a3601e48158a.png)
Epoch#20:
![Epoch#19](https://user-images.githubusercontent.com/41025885/131232520-4175f512-ce89-45b6-9464-5055082e7ccf.png)


This GAN suffers from mode-collapse. 

In order to prevent from the generator to optimizing for a single fixed discriminator, 
*Unrolled GANs* use a generator loss function that incorporates not only the current discriminator's classifications, but also the outputs of future discriminator versions. 

So the generator can't over-optimize for a single discriminator.

Following is the FID graph: (Sorry for not finding the graph with 20 epochs)

![FID loss vs Epochs](https://user-images.githubusercontent.com/41025885/131232734-1f2bf977-dd45-4a65-8e28-38a8056d98cb.png)

Thanks for reading, 

Ziv
