# GAN-FID-fashion-MNIST
Attached is a basic GAN from MNIST FASION image set.

Since the discriminator also improves during training, I needed an *external tool* to evaluate the improvement of the genreted images.

To do this I used FID.
https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance!
[FID loss vs Epochs](https://user-images.githubusercontent.com/41025885/131232480-3b52b685-b663-4bf4-a3e8-ea6ffd3736d5.png)

Following are fewpictures to illustrate the training progress from epoch#0 to epoch#20.

![Epoch#0](https://user-images.githubusercontent.com/41025885/131232494-e1ad44c1-a9c8-419c-9aa2-e6e07921ece8.png)
![Epoch#3](https://user-images.githubusercontent.com/41025885/131232496-898ba6b7-26a4-4011-bd9d-5edf40ed3ed7.png)
![Epoch#10](https://user-images.githubusercontent.com/41025885/131232519-c92ad681-893d-4817-aa2e-a3601e48158a.png)
![Epoch#19](https://user-images.githubusercontent.com/41025885/131232520-4175f512-ce89-45b6-9464-5055082e7ccf.png)

This GAN suffers from mode-collapse. 

In order to prevent from the generator to optimizing for a single fixed discriminator, 
*Unrolled GANs* use a generator loss function that incorporates not only the current discriminator's classifications, but also the outputs of future discriminator versions. 

So the generator can't over-optimize for a single discriminator.

Thanks for reading, 
Ziv
