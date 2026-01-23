I'm doing a project where I want to train a model that "understands" biology by making interpretable predictions. How I intend to do this is as follows: suppose I want the model to understand myeloid leukemia, then:
    1. I train a SAE on the activations of the i-th layer of a single cell foundation model coming from a wide dataset related to myeloid leukemia cells. In this case, I'm doing it on the whole CELLxGENE bone marrow dataset.
    2. Do feature interpretation on the trained SAE by making forward passes of the dataset and check which groups of genes activate each feature, then do some kind of clustering and gene ontology enrichment on the groups
    3. With the scFM and the SAE frozen, I want to do interpretable perturbation prediction on the Replogle (K562) dataset. I will be doing this by training a small MLP adapter that tells me which features from the SAE to steer on the scFM to achieve the perturbed expression. That is, if I trained the SAE on the 10th layer of the scFM, then:

    x^10 = x^10 + s * W_dec 
    where x^10 is the output from the scFM's 10th layer, s = MLP(x^10, perturbation_id) and W_dec is the SAE decoder (which has the features as rows).
    Then I'll let the scFM continue its forward pass and minimize MSE between its output and the perturbed gene expression, only optimizing for the MLP parameters.

    Afterwards, I can look at the resulting vector s which will tell me how did the model need to steer the control expression using interpretable features (something like, more of X pathway, less of Y cell cycle feature, etc) to reach the perturbed state, and I will have an interpretable perturbation prediction model.

I've already downloaded to disk the CELLxGENE bone marrow dataset (split randomly in 19 chunks), which consists of ~900k cells measuring ~38k genes, and the Replogle dataset, which consists of ~2M cells over ~8k genes.

For the scFM, I'm thinking of AIDO.Cell and scBERT, since both are whole transcriptome models, which eliminates the need to introduce data biases by no filtering for HVG genes (like scGPT). Also both models pass all genes through the transformer layers (unlike scFoundation, which only passes non-zero expressed genes), which is absolutely necessary for my steering objective.

My hardware is a single RTX-3090 with 24GB VRAM, but I work in a big cluster which has 70GB RAM.

I've already set up an environment for AIDO.cell and downloaded it (everything related to it is in /ModelGenerator). Did a test run on PBMC3K by linear probing the cell types in the dataset, resulting in good accuracy.

I plan to do the same with scBERT (already have a dedicated environment for it), which I've downloaded at /scBERT. I need you to help me a bit with scBERT since I'm unsure about preprocessing steps.

Here are the repo links for both AIDO.cell and scBERT, respectively.
AIDO.cell: https://github.com/genbio-ai/ModelGenerator/tree/main
scBERT: https://github.com/TencentAILabHealthcare/scBERT

The AIDO.cell repo is actually a bigger repo with more models. A huggingface implementation of AIDO.cell is in https://github.com/genbio-ai/ModelGenerator/tree/main/huggingface/aido.cell 