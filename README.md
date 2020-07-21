# X-SPELLS

<h4>XSPELLS - eXplaining Sentiment Prediction by generating ExempLars in the Latent Space</h4>

Explaining Sentiment Classification with Synthetic Exemplars and Counter-Exemplars

The recent years have witnessed a rapid increase in the use of machine learning models in a wide range of application fields, including businesses, self-driving cars, medicine, public policy, and many others. A large part of those machine learning models are black boxes, i.e., their overall functioning and the logic behind their decisions for a given input instance are not clearly understandable to humans. We present XSPELLS, a model-agnostic local approach for explaining the decisions of a black box model for sentiment classification of short texts. The explanations provided consist of a set of exemplar sentences and a set of counter-exemplar sentences. The former are examples classified by the black box with the same label as the text to explain. The latter are examples classified with a different label (a form of counterfactuals). Both are close in meaning to the text to explain, and both are meaningful sentences, albeit they are synthetically generated. XSPELLS generates neighbours of the text to explain in a latent space using Variational Autoencoders for encoding text and decoding latent instances. A decision tree is learned from randomly generated neighbours, and used to drive the selection of the exemplars and counter-exemplars.

Keras implementation of VAE based on code in [alexeyev's repo](https://github.com/alexeyev/Keras-Generating-Sentences-from-a-Continuous-Space).
