# LymphSeg Project

This repository includes the code for the lymphoma Segmentation Project (lymphSeg).

## Transforming the data to Bids format.
the data was registered via an fsl tool with pydra and the following template. 
The code for the registration can be found in 	[Registration.ipynb](Data_Curation/Registration.ipynb)

<<to be documented >>
Lymphoma BIDS Format (still missing 3 patients from P2 due to issues- ask lucia for new annotations) 
```
Lymphoma BIDS Format  (still missing 3 patients from P2 due to issues- ask lucia for new annotations)

.
├── derivative
	│ └── manual_segm
		│ └── sub-1075196
			│ └── ses-M000
				│ └── anat
					│ └── sub-1075196_ses-M000_dseg.nii.gz
└── sub-1075196
	└── ses-M000
		└── anat
			├── sub-1075196_ses-M000_T1w.json
			└── sub-1075196_ses-M000_T1w.nii
├── dataset_description.json
├── description ( contains all meta-data of the patients regarding age sex acquisition times …)
	-   Age and sex list of the dataset 
	-   Acquisition list of the dataset

```

## Metadata and splits:
We partition the data into four groups depending on their characteristics provided by a medical expert.
The meta-data for the entire lymphoma dataset could be found in 
[participants_.tsv](/Users/rosana.eljurdi/Datasets/Lymphoma/participants_.tsv). The information was generated via the following script 
[Generate_meta_data](/Users/rosana.eljurdi/PycharmProjects/LymphSeg1/DataPreprocessing/Generate_meta_data.py) which grouped information 
extracted from the dicom files and provided by the medical experts while annotating. 
The dataset was split into 4 parts ( (D1) (D2, D3, D4)) according to two createria: the difficulty of the segmentation and the presence of the artefact. D1 refers to clean and easy segmentation, D2 refers to easy  with artefacts. D3 refers tonot easy without artefacts and D4 refers to not easy with artefacts. The classification of the task into easy vs hard was done by a medical expert. To generate the training and testing data, we use the following percentage partitions 70% training and 30% testing. We apply these ratios to each partition 1 and partition 3 representing the MRI data without artefacts (both hard and easy). A third dataset of artefacts was left out as the second testing set. 




## Splitting the dataset 
via  [Lymphoma_CREATESPLITS_Dataset](/Users/rosana.eljurdi/PycharmProjects/LymphSeg1/dataloader.py). 

a train_description.json file is generated with the corresponding train, validation, test and testart datasets. 

## Converting to 2D 

Extract_slices.py for extracting slices and saving them in .npy files

## Training

### Data Loaders

### 3D Unet model



### Inference
This script [inference.py] will compute the 3d predictions of each patient.

## Rename a file

You can rename the current file by clicking the file name in the navigation bar or by clicking the **Rename** button in the file explorer.

## Delete a file

You can delete the current file by clicking the **Remove** button in the file explorer. The file will be moved into the **Trash** folder and automatically deleted after 7 days of inactivity.

## Export a file

You can export the current file by clicking **Export to disk** in the menu. You can choose to export the file as plain Markdown, as HTML using a Handlebars template or as a PDF.


# Synchronization

Synchronization is one of the biggest features of StackEdit. It enables you to synchronize any file in your workspace with other files stored in your **Google Drive**, your **Dropbox** and your **GitHub** accounts. This allows you to keep writing on other devices, collaborate with people you share the file with, integrate easily into your workflow... The synchronization mechanism takes place every minute in the background, downloading, merging, and uploading file modifications.

There are two types of synchronization and they can complement each other:

- The workspace synchronization will sync all your files, folders and settings automatically. This will allow you to fetch your workspace on any other device.
	> To start syncing your workspace, just sign in with Google in the menu.

- The file synchronization will keep one file of the workspace synced with one or multiple files in **Google Drive**, **Dropbox** or **GitHub**.
	> Before starting to sync files, you must link an account in the **Synchronize** sub-menu.

## Open a file

You can open a file from **Google Drive**, **Dropbox** or **GitHub** by opening the **Synchronize** sub-menu and clicking **Open from**. Once opened in the workspace, any modification in the file will be automatically synced.

## Save a file

You can save any file of the workspace to **Google Drive**, **Dropbox** or **GitHub** by opening the **Synchronize** sub-menu and clicking **Save on**. Even if a file in the workspace is already synced, you can save it to another location. StackEdit can sync one file with multiple locations and accounts.

## Synchronize a file

Once your file is linked to a synchronized location, StackEdit will periodically synchronize it by downloading/uploading any modification. A merge will be performed if necessary and conflicts will be resolved.

If you just have modified your file and you want to force syncing, click the **Synchronize now** button in the navigation bar.

> **Note:** The **Synchronize now** button is disabled if you have no file to synchronize.

## Manage file synchronization

Since one file can be synced with multiple locations, you can list and manage synchronized locations by clicking **File synchronization** in the **Synchronize** sub-menu. This allows you to list and remove synchronized locations that are linked to your file.


# Publication

Publishing in StackEdit makes it simple for you to publish online your files. Once you're happy with a file, you can publish it to different hosting platforms like **Blogger**, **Dropbox**, **Gist**, **GitHub**, **Google Drive**, **WordPress** and **Zendesk**. With [Handlebars templates](http://handlebarsjs.com/), you have full control over what you export.

> Before starting to publish, you must link an account in the **Publish** sub-menu.

## Publish a File

You can publish your file by opening the **Publish** sub-menu and by clicking **Publish to**. For some locations, you can choose between the following formats:

- Markdown: publish the Markdown text on a website that can interpret it (**GitHub** for instance),
- HTML: publish the file converted to HTML via a Handlebars template (on a blog for example).

## Update a publication

After publishing, StackEdit keeps your file linked to that publication which makes it easy for you to re-publish it. Once you have modified your file and you want to update your publication, click on the **Publish now** button in the navigation bar.

> **Note:** The **Publish now** button is disabled if your file has not been published yet.

## Manage file publication

Since one file can be published to multiple locations, you can list and manage publish locations by clicking **File publication** in the **Publish** sub-menu. This allows you to list and remove publication locations that are linked to your file.


# Markdown extensions

StackEdit extends the standard Markdown syntax by adding extra **Markdown extensions**, providing you with some nice features.

> **ProTip:** You can disable any **Markdown extension** in the **File properties** dialog.


## SmartyPants

SmartyPants converts ASCII punctuation characters into "smart" typographic punctuation HTML entities. For example:

|                |ASCII                          |HTML                         |
|----------------|-------------------------------|-----------------------------|
|Single backticks|`'Isn't this fun?'`            |'Isn't this fun?'            |
|Quotes          |`"Isn't this fun?"`            |"Isn't this fun?"            |
|Dashes          |`-- is en-dash, --- is em-dash`|-- is en-dash, --- is em-dash|


## KaTeX

You can render LaTeX mathematical expressions using [KaTeX](https://khan.github.io/KaTeX/):

The *Gamma function* satisfying $\Gamma(n) = (n-1)!\quad\forall n\in\mathbb N$ is via the Euler integral

$$
\Gamma(z) = \int_0^\infty t^{z-1}e^{-t}dt\,.
$$

> You can find more information about **LaTeX** mathematical expressions [here](http://meta.math.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference).


## UML diagrams

You can render UML diagrams using [Mermaid](https://mermaidjs.github.io/). For example, this will produce a sequence diagram:

```mermaid
sequenceDiagram
Alice ->> Bob: Hello Bob, how are you?
Bob-->>John: How about you John?
Bob--x Alice: I am good thanks!
Bob-x John: I am good thanks!
Note right of John: Bob thinks a long<br/>long time, so long<br/>that the text does<br/>not fit on a row.

Bob-->Alice: Checking with John...
Alice->John: Yes... John, how are you?
```

And this will produce a flow chart:

```mermaid
graph LR
A[Square Rect] -- Link text --> B((Circle))
A --> C(Round Rect)
B --> D{Rhombus}
C --> D
```