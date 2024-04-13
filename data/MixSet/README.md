# Mixset Dataset Overview

The Mixset dataset comprises a total of 12 JSON files, each containing 300 pieces of MixText data. This brings the total to 3,600 pieces of MixText data across the dataset.

## Dataset Structure

- **Total JSON Files**: 2
- **Total Data Points**: 3,600 MixText data pieces

## Train Test Split

For our Mixset dataset, we have designated the first 250 entries as the training set and the remaining 50 entries as the test set, which is `./MixSet_train.json` and `./MixSet_test.json` respectively. Given that we have both pre-modified and post-modified sentences, we also include the pre-modified sentences in the training set. Therefore, if using the entire Mixset dataset, one would obtain a training set consisting of 3,000 entries and a test set comprising 600 entries. It is important to note that Table 3 in the article only lists the participation of Mixcase in the training set, which may lead to some misunderstandings. 

## Usage

You can directly use the train/test set to evaluate your MGT detector in the binary classification setting. You can also use our MixText setting, which considers the AI-revised HWT and Human-revised MGT as the third class.

### Data Format

The dataset is divided into two main categories based on the source of text rewriting. We have combined all seperated_files together for a better file IO process. Here is the detailed structure of a sample:

```json
{
        "category": "speech",
        "id": 3000,
        "original": "Two twin domes, two radically opposed design cultures. One is made of thousands of steel parts, the other of a single silk thread. One is synthetic, the other organic. One is imposed on the environment, the other creates it. One is designed for nature, the other is designed by her.michelangelo said that when he looked at raw marble, he saw a figure struggling to be free. The chisel was michelangelo's only tool. But living things are not chiseled. They grow. And in our smallest units of life, our cells, we carry all the information that's required for every other cell to function and to replicate.tools also have consequences. At least since the industrial revolution, the world of design has been dominated by the rigors of manufacturing and mass production.",
        "revised": "Two identical domes, two fundamentally different design philosophies. One consists of thousands of metallic components, while the other is crafted from a singular silken strand. One is artificial, the other natural. One imposes itself upon the environment, while the other is created by it. One is designed for nature, the other is designed by Nature herself. Michelangelo believed that when he gazed upon unhewn marble, he witnessed a form yearning to be liberated. His chisel served as the instrument to release its inherent beauty. However, living entities do not conform to such sculptural constraints; they develop and flourish through growth. Within our minuscule biological building blocks, known as cells, resides all the necessary knowledge for their own operation and propagation, as well as that of every other cell. Tools possess implications beyond their intended use. Since the advent of the Industrial Revolution, the realm of design has been governed by the principles of fabrication and large-scale production.",
        "mixset_category": "2llama_polish_token",
        "binary": "MGT"
    }
```
- `category` refers to the category of the original HWT and MGT datasets.
- `id` refers to the unique ID in our newly constructed dataset.
- `original` refers to the original sentence before revising, it is sampled from the pre-existed HWT and MGT datasets.
- `revised` indicates that the Human-revised or LLM-revised sentences contributed by our work.
- `mixset_category` refers to LLM/Human + Operation category defined in our paper, which is detailed in Tables 2 and 3 in our paper.
- `binary` means the category of revised sentences in the binary classification setting.
- 
## Separated_files
We also combine separated files which are evaluate in our code in our dataset.
