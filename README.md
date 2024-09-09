
### Datasests
* For the ELLIPSE and ASAP++ datasets we used in the paper, see .\dataset.

* Chatgpt_essays.txt in .\dataset\ELLIPSE is the essays of chatgpt_data.csv, which can be simply pasted as the input of ChatGPT


### Rerequisites
Before starting the training, please ensure that you have prepared the necessary pretrained model.

* Download deberta-v3-base from HuggingFace https://huggingface.co/microsoft/deberta-v3-base and put the downloaded model in the ./pretrained_models/ directory.

### Reproducing Results
To reproduce the results from the paper:

* Download the checkpoints from here [ell_ckpt](https://drive.google.com/file/d/1Q7OOhHw-xsoJJNREbc4TaAlTlcf240sK/view?usp=sharing) and place them in the `./ckpt` directory.
Run the following command to start inference on the ELLIPSE dataset:
```bash
bash run.sh -t ell -i false
```


### Citation
If you find this code useful, please consider citing our paper:
```
@inproceedings{chen2024multi,
  title={A Multi-task Automated Assessment System for Essay Scoring},
  author={Chen, Shigeng and Lan, Yunshi and Yuan, Zheng},
  booktitle={International Conference on Artificial Intelligence in Education},
  pages={276--283},
  year={2024},
  organization={Springer},
  url={https://doi.org/10.1007/978-3-031-64299-9_22} 
}
``` 