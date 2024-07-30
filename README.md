## XCoOp: Explainable Prompt Learning for Computer-Aided Diagnosis via Concept-guided Context Optimization

Utilizing potent representations of the large vision-language
models (VLMs) to accomplish various downstream tasks has attracted
increasing attention. Within this research field, soft prompt learning has become a representative approach for efficiently adapting VLMs such as CLIP, to tasks like image classification. However, most existing prompt learning methods learn text tokens that are unexplainable, which cannot satisfy the stringent interpretability requirements of Explainable Artificial Intelligence (XAI) in high-stakes scenarios like healthcare. To address this issue, we propose a novel explainable prompt learning framework that leverages medical knowledge by aligning the semantics of images, learnable prompts, and clinical concept-driven prompts at multiple granularities. Moreover, our framework addresses the lack of valuable concept annotations by eliciting knowledge from large language models and offers both visual and textual explanations for the prompts. Extensive experiments and explainability analyses conducted on various datasets, with and without concept labels, demonstrate that our method simultaneously achieves superior diagnostic performance, flexibility, and interpretability, shedding light on the effectiveness of foundation models in facilitating XAI.

### Setup
Use the following command to install the required dependencies:
```
pip install -r requirements.txt
```
This implementation uses Pneumonia dataset as an example, you can download the dataset [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). You can adapt this methods to your datasets by creating new config files in the format of the provided examples (yaml). Remember to also create the `[dataset_name].py` under the `/datasets` folder and import them in `train.py`. 

### Usage
 Use the following command to train the model:
 ```
 python train.py \
 --config-file [config_file_name] \
 --dataset-config-file [dataset_config_file_name] \
 --trainer XCoOp \
 --root [dataset_root_path_name] \
 --output-dir [output_directory_path_name] \
 --resume false
 ```
 
 Please see `train_script.sh` for an example.

 ### Citation
 ```
@article{bie2024xcoop,
  title={XCoOp: Explainable Prompt Learning for Computer-Aided Diagnosis via Concept-guided Context Optimization},
  author={Bie, Yequan and Luo, Luyang and Chen, Zhixuan and Chen, Hao},
  journal={arXiv preprint arXiv:2403.09410},
  year={2024}
}
 ```

 ### Acknowledgement
 Our code is based on [CoOp](https://github.com/KaiyangZhou/CoOp) and [LASP](https://github.com/1adrianb/lasp). Thanks for their amazing works.