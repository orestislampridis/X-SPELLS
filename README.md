# X-SPELLS - eXplaining Sentiment Prediction by generating ExempLars in the Latent Space
**Orestis Lampridis**, Aristotle University of Thessaloniki, Greece, lorestis@csd.auth.gr  
**Riccardo Guidotti**, Department of Computer Science, University of Pisa, Italy, riccardo.guidotti@unipi.it   
**Salvatore Ruggieri**, Department of Computer Science, University of Pisa, Italy, salvatore.ruggieri@unipi.it	

The recent years have witnessed a rapid increase in the use of machine learning models in a wide range of application fields, including businesses, self-driving cars, medicine, public policy, and many others. A large part of those machine learning models are black boxes, i.e., their overall functioning and the logic behind their decisions for a given input instance are not clearly understandable to humans. We present XSPELLS, a model-agnostic local approach for explaining the decisions of a black box model for sentiment classification of short texts. The explanations provided consist of a set of exemplar sentences and a set of counter-exemplar sentences. The former are examples classified by the black box with the same label as the text to explain. The latter are examples classified with a different label (a form of counterfactuals). Both are close in meaning to the text to explain, and both are meaningful sentences, albeit they are synthetically generated. XSPELLS generates neighbours of the text to explain in a latent space using Variational Autoencoders for encoding text and decoding latent instances. A decision tree is learned from randomly generated neighbours, and used to drive the selection of the exemplars and counter-exemplars.

Keras implementation of VAE based on code in [alexeyev's repo](https://github.com/alexeyev/Keras-Generating-Sentences-from-a-Continuous-Space).

An improved version of X-SPELLS is available [here](https://github.com/lstate/X-SPELLS-V2).

## License

MIT License for the source code in the lstm_vae directory. <br />
BSD 2-Clause "Simplified" License for the source code in the lime directory. <br />
Apache-2.0 License for the rest of the project.

## Citation

O. Lampridis, R. Guidotti, S. Ruggieri. [Explaining Sentiment Classification with Synthetic Exemplars and Counter-Exemplars](https://doi.org/10.1007/978-3-030-61527-7_24). Discovery Science (DS 2020). 357-373. Vol. 12323 of LNCS, Springer, September 2020. <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Open_Access_logo_PLoS_transparent.svg/220px-Open_Access_logo_PLoS_transparent.svg.png" alt="Excel Sheet" width="10"/> 

## References

Bowman, S.R., Vilnis, L., Vinyals, O., Dai, A.M., Jozefowicz, R. and Bengio, S., 2015. Generating sentences from a continuous space. arXiv preprint arXiv:1511.06349.
