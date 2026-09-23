# ORCA-OCT-Classifier

## 📝 Short introduction

Hi, I'm Susanna! This is my Bachelor's thesis in Artificial Intelligence (the joint degree between the University of Pavia, the University of Milan and Milano Bicocca), and it comes from something I care about a lot: using AI in medicine, and in this case in ophthalmology.

The project started in the Medical Devices lab, where I began working on it together with my classmate Carlotta Zanei. I then kept going on my own and turned it into my thesis, with Professor Giovanna Nicora as my supervisor.

**ORCA (OCT Retinal Classification Assistant)** is a decision support system that looks at retinal OCT scans and sorts them into three categories: Age related Macular Degeneration (AMD), Diabetic Macular Edema (DME) and Normal. These are two of the leading causes of vision loss worldwide, and clinics produce far more scans every day than there are specialists to read them.

The heart of the project is a comparison between two kinds of approaches. On one side there is "classic" deep learning: a small CNN built from scratch and EfficientNet adapted with transfer learning. On the other side there are two recent medical foundation models, MedGemma and MIRAGE.

Because accuracy alone is not enough for a tool that could end up near real patients. So for every model I also looked at whether its confidence can be trusted (calibration), whether it can realize when it is looking at something it has never seen, like glaucoma, instead of forcing it into a known class (out of distribution detection), and whether we can see where it is looking when it makes a decision (explainability with heatmaps).

> ⚠️ **Important:** ORCA is a research demonstrator, not a clinical product. It does not replace the judgement of a qualified ophthalmologist.


## 🛠️ Technologies/tools used

* **Python** for everything, from data preparation to the web app.
* **Kaggle Notebooks** (GPU T4 x2): where I ran most of the experiments. I started on **Google Colab** (GPU T4), then moved to Kaggle to get more GPU power and more comfortable training sessions.
* **PyTorch and torchvision**: to build, train and evaluate the CNN, EfficientNet and MIRAGE.
* **Hugging Face Transformers**: to load MedGemma and use its vision encoder as a feature extractor.
* **Scikit-learn**: for the patient level data split, class weights and evaluation metrics.
* **Pandas, Matplotlib and Seaborn**: for data analysis and for all the plots in the results (confusion matrices, ROC curves, reliability diagrams).
* **Grad-CAM**: to generate the saliency maps that show where the convolutional models are looking.
* **Streamlit**: to build the ORCA prototype, the interactive interface where you can upload a scan and try the models.


## 🧩 Models compared

I compared four approaches, going from classic deep learning to the newest foundation models:

* **CustomCNN**: a small convolutional network built from scratch. It is my baseline, to see how far you can get without any pretraining.
* **EfficientNet-B0**: a CNN pretrained on ImageNet and adapted to OCT with transfer learning.
* **MedGemma-4B**: Google's medical foundation model. I kept its vision encoder frozen and trained only a small classifier on top of the features.
* **MIRAGE-Base**: a Vision Transformer pretrained specifically on retinal OCT. I first trained only the classification head (linear probing) and then fine tuned the whole model.


## ✨ Features

* **Three class classification** (AMD, DME, Normal) with a patient level split, so that scans from the same person never end up in both training and test. This avoids data leakage and inflated results.
* **Calibration** with temperature scaling, so that the confidence of a model actually reflects how often it is right.
* **Out of distribution detection** using entropy, tested on glaucoma images that the models never saw during training.
* **Visual explanations**: Grad-CAM for the convolutional models and occlusion sensitivity for MIRAGE.
* **Interactive prototype** built with Streamlit, with uncertainty estimation through Monte Carlo Dropout.

## 🚀 Process

I started from OCT5k, a public retinal OCT dataset from UCL. Since my goal was classification and not segmentation, I used the whole collection of raw scans, and the first job was cleaning it. Out of 3,348 files, 117 turned out to be colour fundus photographs and not OCT scans, so I removed them. That left me with 3,231 images from 117 patients.

Then I built the preprocessing pipeline (grayscale, resizing to 512 x 512, normalisation) and added data augmentation on the training set only. The step I care about most is the split. Every patient has many scans, around 27 on average, so I split the data at patient level: 77 patients for training, 21 for validation and 19 for test. Nobody appears in more than one set, which keeps the results honest.

After that I trained the four models with a class weighted loss, because AMD is only 15% of the data. Then came the part that makes ORCA more than a classifier: temperature scaling for calibration, entropy scoring to catch out of distribution images (I used glaucoma scans from the SYNOCT dataset), and heatmaps to see where the models look.

Finally I put everything into the Streamlit prototype. I also looked at ORCA from the regulatory and ethical side, to understand what it would take to become a real medical device in the EU (SaMD) and how it holds up against the FUTURE AI guidelines.


## 📊 Results at a glance

All numbers are computed on the held out test set.

| Model | Accuracy | Macro AUC | AMD recall | OOD AUROC (glaucoma) |
|---|---|---|---|---|
| CustomCNN | 0.854 | 0.917 | 0.333 | 0.030 |
| EfficientNet B0 | 0.957 | 0.995 | 0.812 | 0.872 |
| MedGemma 4B* | 0.900 | 0.979 | 0.880 | 0.204 |
| MIRAGE Base | 0.960 | 0.995 | 0.870 | 0.977 |

*MedGemma was evaluated on 511 of the 553 test images, because feature extraction failed on 42 of them.*

The main takeaway is that **no single model wins everywhere**:

* **MIRAGE Base** has the best accuracy and by far the best out of distribution detection, but it is very overconfident and needs a large temperature correction (T = 4.773) to be usable.
* **EfficientNet B0** is the most balanced one: near top classification, good OOD detection and clear Grad-CAM explanations.
* **MedGemma 4B** has the highest AMD recall even with a frozen encoder, but it labels every glaucoma image as AMD with high confidence, so it does not notice that anything is wrong.
* **CustomCNN** is the baseline, and it shows why pretraining matters: it struggles with AMD and calls every glaucoma image DME with almost 99% confidence.
  
### A few figures

<img width="2094" height="1906" alt="confusion_matrices" src="https://github.com/user-attachments/assets/6221eb14-60ad-4687-88ac-8c32ccfbcc44" />

*Confusion matrices on the test set. The CustomCNN mixes up more than half of the AMD scans with DME, while the other models are much more consistent.*

<img width="2074" height="1753" alt="ood_entropy_histograms" src="https://github.com/user-attachments/assets/3e294e3f-6f3b-47d6-bb82-e4b74e40cb17" />

*Entropy of the predictions on in distribution images versus glaucoma images. MIRAGE and EfficientNet push glaucoma towards high entropy, so it gets flagged. CustomCNN and MedGemma stay confident and miss it.*

<img width="1790" height="914" alt="gradcam_test_samples" src="https://github.com/user-attachments/assets/3db2f89e-3d4a-4dcd-8edc-ef54188fd2ad" />

*Grad-CAM heatmaps for CustomCNN (top) and EfficientNet (bottom). Warmer colours mark the regions that pushed the model towards its prediction.*


## 🧠 What I learned

This project taught me more than any other I have done so far, and not only about code.

The biggest lesson is that a model with great accuracy is not automatically a good model. Before this thesis I would have picked the winner by looking at one number. Now I know that accuracy, calibration, out of distribution safety and explainability are different questions, and a model can be excellent at one and terrible at another. MIRAGE was the best classifier but needed a huge temperature correction to stop being overconfident. MedGemma had the highest AMD recall but did not notice that glaucoma was something new. 

On the technical side, I learned how to build an evaluation that can be trusted, how to work with pretrained and foundation models, and how to calibrate a model and detect inputs it should not be answering. I also learned to be honest about limits: my explanations are only a prompt for a clinician to check, and my results come from one dataset and one scanner.

The clinical, regulatory and ethical part surprised me the most. I started thinking about it as an "extra" chapter, and I ended up finding it just as important as the models. Looking at ORCA as a possible medical device, with rules, risk classes and documents I did not have, made me realise how far a working prototype is from something that could be used on real patients. Thinking about automation bias, privacy and who is responsible when the system misses something changed the way I look at AI in healthcare: building the model is only one piece of the job.


## 🔧 How could it be improved

ORCA is a prototype, and there is a lot I would still like to do. If I continue working on it, I would love to:

* **Test it on other scanners and hospitals.** Everything here comes from one dataset acquired with one type of scanner (Heidelberg Spectralis). I want to know if the results still hold on images from other devices and other centres, because that is where models usually start to struggle.
* **Use data with patient information.** OCT5k has no age, sex or other details, so I could not check whether the models work worse for some groups of people. A dataset with this metadata would make a proper fairness analysis possible.
* **Try many more unseen pathologies.** I tested the out of distribution detection only on glaucoma. I am curious to see what happens with conditions like macular hole or epiretinal membrane, because I expect the four models to behave very differently.
* **Ask real ophthalmologists.** My heatmaps look reasonable to me, but I am not a clinician. A usability study would show if the explanations and the uncertainty alerts actually help in a real workflow, or if they just make people trust the tool too much.
* **Explain MedGemma too.** Since its encoder is frozen, I could not produce heatmaps for it. Methods like LIME or SHAP could give it a visual explanation as well, so that all four models can be compared on this side too.


## 📁 Repository structure

* **notebooks**: one notebook per modelling approach. The first one covers both the CustomCNN and EfficientNet B0, the second one MedGemma 4B and the third one MIRAGE Base. Each notebook goes from data loading to training, calibration, OOD detection and explanations.
* **orca-app**: the code of the Streamlit prototype.
* **requirements.txt**: the Python packages you need.
* **Thesis and slides**: the full written work and the presentation I used for the defence.
* **Model weights**: they are too big for the repository, so you can find them in the [Releases](../../releases) section of this page. You need them to run the app.

### 📱 Try the app

Scan the QR code to open the ORCA prototype, or use the link below.

<img width="574" height="574" alt="qr-code" src="https://github.com/user-attachments/assets/e1854d5a-d2ec-4cbd-926e-78aec6a56e76" />

[Open ORCA](https://orca-oct-classifier-dlxltwcshusintuhfq89ak.streamlit.app/)

### 📚 Datasets and papers

* **OCT5k** (training and evaluation): retinal OCT scans of AMD, DME and healthy eyes. [Dataset](https://rdr.ucl.ac.uk/articles/dataset/OCT5k_A_dataset_of_multi-disease_and_multi-graded_annotations_for_retinal_layers/22128671?file=44436359) and [paper](https://www.nature.com/articles/s41597-024-04259-z) by Arikan et al.
* **SYN OCT** (out of distribution test): synthetic OCT images of healthy and glaucoma eyes, from Wong et al. [Dataset](https://zenodo.org/records/17151869) and [paper](https://www.nature.com/articles/s41597-026-06946-5).
* **MIRAGE**: the foundation model behind MIRAGE Base, by Morano et al. [Code and weights]([https://github.com/j-morano/MIRAGE](https://github.com/j-morano/MIRAGE)).
* **MedGemma**: Google's medical foundation model. [Model page on Hugging Face](https://huggingface.co/google/medgemma-4b-it).

The datasets are not included in this repository. Please download them from the links above and follow the instructions in the notebooks.


## 📄 Full thesis

This README only scratches the surface. If you want the full story, including the design choices, the details of each model, all the metrics, the calibration and OOD analysis, and the regulatory and ethical discussion, please read the complete thesis: [`ORCA_thesis.pdf`](ORCA_thesis.pdf).

<br>

---

**Susanna Mazzocchi**
Bachelor in Artificial Intelligence, University of Pavia, University of Milan and Milano Bicocca
Supervisor: Prof. Giovanna Nicora
Academic Year 2025/2026
