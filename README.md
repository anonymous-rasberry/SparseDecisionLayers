# Supplementary material for "Leveraging Sparse Linear Layers for Debuggable Deep Networks"

Contents: 
+ `supplementary.pdf` is the full paper with the appendix
+ `elasticnet.py` contains the core SAGA-based solver for fitting elastic net regularized generalized linear models at scale
+ `imagenet_features.py` precomputes deep feature representations for the ImageNet dataset 
+ `imagenet_example.py` fits the regularization path for ImageNet
+ `nlp_examples.py` fits the regularization path for the NLP tasks (toxic comment and sentiment classification)
+ `vision_examples.py` fits the regularization path for smaller vision datasets (Places10)
+ `nlp_evaluate.py`, `nlp_modeling.py`, and `nlp_dataset.py` are modified helper files for processing the NLP tasks from `https://github.com/barissayil/SentimentAnalysis`. 


Dependencies: 
+ `torch`
+ `robustness` library from `https://github.com/MadryLab/robustness` 
+ `cox` library from `https://github.com:MadryLab/cox.git` 
+ `transformers` library from `https://huggingface.co/transformers/`
+ `datasets` library from `https://github.com/huggingface/datasets`
