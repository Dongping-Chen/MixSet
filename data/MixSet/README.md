# Mixset Dataset Overview

The Mixset dataset comprises a total of 12 JSON files, each containing 300 pieces of MixText data. This brings the total to 3,600 pieces of MixText data across the dataset.

## Dataset Structure

- **Total JSON Files**: 12
- **Total Data Points**: 3,600 MixText data pieces

### Data Format

The dataset is divided into two main categories based on the source of text rewriting:

1. **HWT Rewritten Data (by llama and GPT)**:
   - Each JSON file contains entries with two keys:
     - `HWT_sentence`: The original sentence before rewriting.
     - `xxx_output`: The modified sentence after being rewritten by llama or GPT models.

2. **Human Revised MGT Data**:
   - This category includes human revisions and is split into two sub-categories, humanize and MGT_polish, with four files each.
   - For these files, the JSON structure contains:
     - `MGT_sentence`: The original sentence.
     - `xxx_output`: The modified sentence after human revision.

## Key Naming Convention

- `xxx_output` refers to the output from either the automated models (llama and GPT for HWT data) or the human revision process (for MGT data).
- The prefix (`HWT_` or `MGT_`) indicates the source or type of the original sentence.

## Usage

This dataset is intended for use in natural language processing research and development, offering a diverse set of rewritten texts for various applications such as text augmentation, machine translation quality evaluation, and more.

