# An Adversarial Perspective on "Overinterpretation Reveals Image Classification Model Pathologies"

Jacob Springer — Jan 12, 2022

## *Abstract*

A [recent paper](https://arxiv.org/abs/2003.08907) by Carter et al., 2021 suggests that neural networks image classifiers *overinterpret* images, meaning that the classifier finds evidence for the correct class label based on non-human-interpretable patterns that are present in small subsets of pixels from the unmodified images. Carter et al. argue this idea by finding these so-called Sufficient Input Subsets (SIS) and demonstrating that images that contain only these subsets are confidently and correctly labeled by the classifier. Since standard neural network image classifiers can only take as input an entire image, an image containing only a subset of pixels must replace the "removed" pixels with a color that is expected to have no affect on the output. The authors choose gray. The authors claim that the high confidence of image classifiers on SIS images arises from the non-human-interpretable but predictive content of the retained pixels in each SIS image. In this article, I will argue that this is not the case. Instead, I will argue that the "removed" (i.e., gray) pixels introduce adversarial-example-like patterns that cause the classifier to confidently classify the input. In fact, I will argue that classifiers do *not* find class-evidence in the retained pixel subsets alone. Finally, I will show that the algorithm proposed by the author can be used to construct adversarial SIS images that are confidently but incorrectly classified.

## I    Introduction

A group at MIT recently released a paper titled [*Overinterpretation reveals image classification model pathologies*](https://arxiv.org/abs/2003.08907) (Carter et al., 2021), arguing that neural network image classifiers can often classify images correctly with only a tiny non-human-interpretable subset of the pixels of the original image. The authors argue that this result supports the idea that neural network classifiers *overinterpret* the input images—that is, the classifier finds evidence for the class label in areas of the image that lack human-interpretable patterns. 

<img title="" src="https://sprin.xyz/assets/blog/overinterpretation/figure1.png" alt="Figure 1" data-align="inline">Figure 1.  Evidence for neural network overinterpretation. This figure is a reproduction of Figure 4 from Carter et al. (2021). This figure shows the original images (top), the Sufficient Input Subset computed by the BGSIS method (middle), and the corresponding mask computed by the Batched Input SIS method (bottom). Each Sufficient Input Subset consists of a sparse set of unmodified pixels from the original image, where the remaining pixels are removed. Each Sufficient Input Subset is confidently classified as the true label (shown above original images). The Sufficient Input Subset images are presented to the classifier exactly as they are shown here, meaning that a "removed" pixel will be set to gray. Each mask indiciates which pixels are included in the Sufficient Input Subset. Purple indicates that a pixel is removed (i.e., set to gray), and yellow indicates that a pixel remains in the Sufficient Input Subset. 

The authors present an algorithm called Batched Gradient SIS (BGSIS) to find an image containing only a sparse set of unmodified pixels from an original image such that the classifier confidently classifies the new image correctly. This is called a Sufficient Input Subset (SIS). Importantly, neural network image classifiers can only take as input an entire image. Thus, a pixel can only be "removed" from the image by setting it to a color which is expected to have no effect on the classifier. A reasonable choice for such a color is the average color across every pixel in the dataset—gray. Thus, each image found by the algorithm consists of the sparse set of unmodified pixels and the remaining pixels which have been set to gray. Examples of SIS images are presented in Figure 1 in the middle row. BGSIS iteratively removes pixels (by setting pixels to gray) that least decrease the confidence. 

The idea that neural networks can find class-evidence from non-interpretable patterns is not new. Neural networks are well-known to be susceptible to adversarial examples (see [Szegedy et al. (2014)](https://arxiv.org/abs/1312.6199), [Goodfellow et al. (2015)](https://arxiv.org/abs/1412.6572), and Figure 2), which are small non-interpretable perturbations to an image which cause the classifier to confidently but incorrectly label the image. Carter et al. argue that, unlike adversarial examples, the existence of confidently-classified SIS images suggest that classifiers find strong class-evidence in the non-interpretable but predictive areas of the original image, rather than added components.

![](https://sprin.xyz/assets/blog/overinterpretation/figure2.png)

In this article, I will argue that this is not the case. I will show that the set of pixels that BGSIS discovers as "sufficient" are often not sufficient, and rather it is the patterns introduced by the "removed" (i.e. gray) pixels that add information to the image to maintain or increase the confidence of a particular class. I will conclude by arguing that the patterns introduced by these removed pixels act like adversarial examples.

## II    The Batched Gradient SIS method does not find sufficient input subsets

Carter et al. argue that the image classifier finds evidence for its classification in the pixels that are identified by BGSIS as a Sufficient Input Subset. Here, I propose a modified experiment to control for the possibility that the "removed" (gray) pixels introduces patterns that might affect the classification score. The goal of my experiment is to re-introduce previously removed pixels in order to get rid of (some of) the patterns that are introduced by the "removed" (gray) pixels in the SIS images. Since I will retain every pixel in the Sufficient Input Subset that was originally identified by BGSIS, we should expect that if the Sufficient Input Subset is sufficient for classification, then the classifier should continue to classify the image correctly with high confidence.

The experiment is as follows: first, I will run BGSIS to find a Sufficient Input Subset, and then, I will "smooth" this subset by reintroducing pixels that are adjacent to pixels that are in the original Sufficient Input Subset. I call the resulting image a *Smoothed Sufficient Input Subset*. Every pixel that is in the original Sufficient Input Subset will be included in the corresponding Smoothed Sufficient Input Subset, so that the fine-detail patterns created by the border of the gray pixels itself cannot contribute to the confidence of the classifier. More precisely, the smoothed masks are constructed by convolving each SIS mask with a 3x3 matrix of 1's, thus removing from each smoothed mask any pixel that is adjacent (including diagonally) to a pixel that was kept by BGSIS.

<img title="" src="https://sprin.xyz/assets/blog/overinterpretation/figure3.png" alt="">

The results of this experiment, shown in Figure 3, demonstrate that while the true-label confidence of the classifier is high for the original and SIS images, the confidence is low for the smoothed pixel subsets is near-zero. The substantially smaller confidence of the classifier for the Smoothed Sufficient Input Subsets suggests that the confidence of the classifier on the Sufficient Input Subsets can be, in large part, attributed to the patterns introduced by the removed (gray) pixels themselves. Furthermore, the Smoothed Sufficient Input Subset images contain every pixel that is included in the corresponding Sufficient Input Subset but are nonetheless misclassified, which implies that the Sufficient Input Subsets are not sufficient for classification.

## III    Gradient-based saliency of sufficient input subsets is concentrated around the border between removed and retained pixels

My next experiment measures the importance, i.e., the saliency, of each pixel to the classifier. If the classifier is finding strong class-evidence on the subset of pixels identified by BGSIS, then saliency methods should identify these pixels as highly salient. I run a saliency method called the SmoothGrad method (Smilkov et al., 2017) to determine pixel saliency of SIS images. I overlay the saliency maps on top of the corresponding pixel masks generated by BGSIS.

<img title="" src="https://sprin.xyz/assets/blog/overinterpretation/figure4.png" alt="">Figure 4. The most salient pixels of Sufficient Input Subset images superimposed over the mask that corresponds to each SIS image. Purple pixels correspond to the masked pixels and yellow pixels correspond with the pixels remaining in the generated sufficient input subset (compare to Figure 1). White semi-transparent pixels correspond with the most salient pixels in the image. The important takeaway from this figure is that the white, highly-salient pixels are concentrated around the boundary between removed and retained pixels from the Sufficient Input Subsets (i.e., the boundary between the yellow and purple pixels), and do not overlap substantially with most of the retained pixels in the Sufficient Input Subsets (i.e., the yellow pixels). Note that while this figure shows images of the saliency superimposed over the mask corresponding with the SIS image, I compute the saliency of SIS images themselves (for examples, see Figure 1, middle row).

The most salient pixels (white semi-transparent pixels in Figure 4) are concentrated around the pixels are on the border between the retained (yellow) and removed (purple) pixels. This is consistent with the hypothesis that the classifier finds evidence for its classification in the high-frequency patterns that are introduced by removing pixels (i.e., introducing gray pixels) from the Sufficient Input Subset. Furthermore, the bulk of the pixels in the Sufficient Input Subset (yellow in Figure 4) are not consider important, suggesting that the pixels retained by BGSIS are not important on their own for classification. This explains the result from the previous section: since the patterns introduced by the border between removed and non-removed pixels are the most salient pixels, removing them by smoothing the border between removed and retained pixels will reduce the classification confidence substantially.

## IV    Sufficient Input Subset images are adversarial examples

The following experiments demonstrate that SIS images generated with the BGSIS algorithm can be thought of as adversarial examples: the BGSIS algorithm introduces patterns into the image by removing pixels that increases the confidence of the true label by way of gradient ascent. Thus, the classifier confidence of the true label of SIS images is often substantially higher than the corresponding classifier confidence on the original image. Furthermore, BGSIS can be modified to increase the classifier confidence of an arbitrary class label, thus creating adversarial SIS images.

First, I present a visualization of the SIS algorithm and the classifier confidence associated with the image at each step. I present the pixel mask, the resulting masked image, and the classifier confidence for each iteration of the algorithm.  BGSIS runs by iterative choosing the best pixels to mask. The algorithm outputs the image with a as many masked pixels as possible so that the confidence remains above the threshold. Since, at every step, BGSIS will mask additional pixels, the algorithm will select the masked image with the highest iteration number where this masked image is classified with above-threshold confidence and that no future iteration of masked image will be classified with above-threshold confidence. 

<center><video class="video-background" autoplay loop muted playsinline width="620px">
<source src="https://sprin.xyz/assets/blog/overinterpretation/figure5.mp4" type="video/mp4" />
</video></center>Figure 5. Animated visualization of the BGSIS algorithm and corresponding classification confidence over each iteration in the algorithm. The first image in each set is the mask at the <i>i</i>th iteration (compare to Figure 1, bottom), the second image is the masked image (compare to Figure 1, center), and the graph to the right is the classifier confidence of the true label (above the graph). The key takeaway from this figure is that the classifier confidence of the true label near-universally increases as the algorithm progresses.

There are two key takeaways from Figure 5:

1. As the algorithm runs, confidence near-universally increases to nearly 100%. While the algorithm is designed to remove information by removing pixels thus lowering confidence over each iteration, it is instead evident that removing pixels increases the confidence, suggesting that the removed pixels introduce patterns that support classification confidence. As discussed above, I suspect that the patterns created at the boundary of the removed pixels increase the classification confidence.
2. Notably the confidence of the classifier on the image of the chime and the image of the barrel starts at nearly 0%. This is the case when the classifier incorrectly classifies the original image. However, Figure 4 shows that the BGSIS method increases the confidence of this image well above the 90% threshold, suggesting that even when patterns supporting a particular class are not present or only minimally present, removing pixels will increase the confidence.

These results suggest that the BGSIS method is essentially a search for adversarial image masks. The algorithm operates by computing the gradient of the confidence with respect to the mask, and then adjusts the mask to minimally decrease (i.e., maximally increase) the confidence. Broadly, this is the same structure as a gradient-based search for an adversarial example. Thus, it is no surprise that this algorithm will act similarly to any adversarial example search algorithm.

The authors include a similar figure to our Figure 5 in the appendix of their paper (Figure S15 from their paper). Their figure illustrates the classifier confidence for each iteration of the algorithm, and includes substantially more data than our Figure 5. Similarly to our Figure 5, they find that the classifier confidence on the true label of each image almost always increases quickly to nearly 100%.

To explore this idea that this method can be used to construct adversarial masks in more depth, I construct explicitly-adversarial SIS images using the BGSIS algorithm that aim to cause the classifier to identify a particular (adversarial) target label with a confidence above 90%. Instead of removing pixels to increase the confidence of the true label of the image, I remove pixels to increase the confidence of the adversarial target label. 

<img title="" src="https://sprin.xyz/assets/blog/overinterpretation/figure5.png" alt="">Figure 6. Class-targeted adversarial SIS images generated using BGSIS. I render pairs of the original image and the confidence on the true label (top) and the adversarial image and the confidence on the adversarial target label (bottom).

These adversarial SIS images have similar properties to the original SIS images from Figure 1: for many of the examples, the confidence of the classifier is above 90%, and the masks cover most of the images. Consistent with our previous experiments, the existance of adversarial SIS images suggests that the masks themselves carry information pertinent to classification.

To illustrate this idea visually, I present (admittedly, somewhat cherry picked) masks generated using adversarially-trained classifiers which are known to have more-human-aligned gradients (Engstrom et al., 2019).

<img title="" src="https://sprin.xyz/assets/blog/overinterpretation/figure6.png" alt="">Figure 7. Original images (top), SIS images (center) and corresponding masks (bottom) generated using BGSIS on an adversarially-trained ResNet50.

The removed pixels form interesting patterns that appear semantic when the classifier is robust (Figure 6). The semantic nature of the mask is not (always) an artifact of outlining a semantic feature present in the original image. For example, the image of the ears of corn are masked to appear as ears of corn that do not align with the original ears of corn. Similarly, the image of the baseball is masked to include the appearence of a larger baseball than appears in the original image. The image of the panda is masked to include the apparence of either whiskers or possibly long thin bamboo-type leaves. While this experiment is less rigorous, it demonstrates visually how removed pixels can form patterns that are meaningful to the classifier.

## V    Removed pixels are real statistical patterns.

The authors find that they can train models on the 5% pixel-subset images that attain minimal accuracy loss compared to the original models and they conclude that the 5% pixel subset has real statistical patterns that correlate with the label well enough on which to base a prediction. I would suspect that a better experiment to determine if this is true would be to train on the smoothed masked images from above, although it is not an experiment that wIwill present here due to the computational cost of training a new model. As an alternative hypothesis, it is possible that the patterns generated by the removed pixels are real statistical patterns that are useful for classification. The idea that you can train a classifier on adversarial examples and yield non-trivial generalization accuracy has been shown by Ilyas et al. (2019) in the paper [*Adversarial Examples Are Not Bugs, They Are Features*](https://arxiv.org/abs/1905.02175). A result suggesting that training on the adversarial patterns introduced by the removed pixels of sufficient input subsets generated by BGSIS would be consistent with Ilyas et al.

## VI    Conclusion

The authors of this paper raise a very important point: be very critical of any attempt to interpret a deep learning model, as they are often subject to the bias of the interpretation algorithm.

I don't think that the results that the authors present are wrong or uninteresting---quite the contrary. It is striking to us that 95% or more of the pixels in an image can be replaced with gray and that a classifier will maintain confidence in its classification. However, unless there is a way to explicitly remove input from the model, the interaction between the removed input and the remaining input, such as the patterns made at the border of the removed grey pixels, will always have a possibility of affecting the behavior of a classifier. 

Perhaps a better approach would be to define a sufficient input subset as a subset of pixels such that, if present, will *always* cause the classifier to confidently predict the label under any perturbation to the other pixels. I doubt this will be useful, firstly because wIwould speculate that such a set would be difficult to compute, and secondly because neural networks are so vulnerable to adversarial examples that such a set would likely consist of nearly every pixel in the original image, so as to not introduce many degrees of freedom that might allow a vulnerability to arise. 

I would speculate that similar approaches to interpretability that aim to determine the contribution of individual pixels will be vulnerable to similar adversarial explanation as is described by this article. Thus, any attempt to interpret a deep learning model should be met with skepticism, as often the bias of the algorithm will shine through.

### Technical details

The code to reproduce our experiments is available at [my GitHub page](https://github.com/jakespringer/overinterpretation).

The classifiers I used were downloaded from [this repository from Microsoft](https://github.com/microsoft/robust-models-transfer). I used the standard ResNet50 (ɛ=0) and an adversarially-trained ResNet50 (ɛ=3).

All images you see in this article are from [ImageNet](https://image-net.org).

## *References*

Carter, B., Jain, S., Mueller, J., & Gifford, D. (2020). Overinterpretation reveals image classification model pathologies. *Advances in Neural Information Processing Systems*, 34.

Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R. (2014). Intriguing properties of neural networks. *2nd International Conference on Learning Representations, ICLR 2014*.

Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. *3rd International Conference on Learning Representations, ICLR 2015*.

Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017). Smoothgrad: removing noise by adding noise. *arXiv preprint arXiv:1706.03825*.

Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Tran, B., & Madry, A. (2019). Adversarial robustness as a prior for learned representations. *arXiv preprint arXiv:1906.00945*.

Salman, H., Ilyas, A., Engstrom, L., Kapoor, A., & Madry, A. (2020). Do adversarially robust imagenet models transfer better?. *arXiv preprint arXiv:2007.08489*.

Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B., & Madry, A. (2019) Adversarial Examples Are Not Bugs, They Are Features, *Advances in Neural Information Processing Systems*, 32.
