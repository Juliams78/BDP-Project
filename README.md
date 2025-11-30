# 🔮 Fake News Detection using Spark + LSTM

## 🎯 Purpose
The primary objective is to replicate the experiment conducted on the paper [**Real-Time Fake News Detection Using Big Data Analytics and Deep Neural Network (IEEE, 2023)**](https://doi.org/10.1109/TCSS.2023.3309704) as 
a requirement for the final project of the discipline Big Data Programming for the Big Data course at Chungbuk National University.

The paper describes a Fake News Detection Model using LSTM and Deep Neural Network, with the intention of using the model on real-time fake news detection in social networks, 
where time is an important feature to avoid the spread of fake-news and misinformation. 

The experiment conducted in this project was conducted in a smaller scale and using different datasets from [Kaggle](https://www.kaggle.com/) in two different environments. The first environment being Google Colab and the other being the 
Docker Desktop installed in a local machine. 

---

## ⚙️ Google Colab
- For the Google Colab implementation the only needed files are the [Colab notebook](https://colab.research.google.com/drive/1NWgd49RnairWfTNXNAdWriSgY-2ssqi3?usp=sharing) and the two folders: Data, with the dataset files, and Results, where the generated files with the results will be stored.
The folders should be in the user's Google Drive, since the notebook mounts the drive during the execution.
- It's important to change the path of the variables **"DATA_DIR"** and **"OUTPUT_DIR"** to the according paths on Cell #2 before execution.

---

## 📦 Local Implementation
Using **Docker + Hadoop + Spark** to run the full pipeline in a fully distributed environment.

- **Environment and Needed Tools**:
  The experiment was conducted in a machine with Windows 11, 32GB RAM and the docker container was built with 6GB (this configuration can be changed in the code, in the spark-session creationmment, if necessary).
  -  Visual Studio (optional)
  - [Docker](https://www.docker.com/)

- **Structure and Needed Files**:
Example of how the folder should be for the code to work.

- `docker-spark/`
  - `docker-compose.yml`
  - `Dockerfile`
  - `requeirements.txt`
  - `README.md` 
  - `scripts/`
    - `bdp_project.py`
    - `Data/`
       - [`fake-and-real-news-dataset/`](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
         - `Fake.csv`
         - `True.csv`
      - [`LIAR-DATASET/`](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset?resource=download)
        - `test.tsv`
        - `train.tsv`
        - `valid.tsv`
    - `Results/`
      - Generated files during the execution
 
- **How to Run**:
  With all the repositories in the correct structure, open the "docker-spark" folder on a terminal on VSCode and execute the following commands:

 
``` docker-compose build ```

``` docker-compose up -d ```

``` docker exec -it spark-master bash ```

Now, already inside the spark-master the following code:

``` /opt/spark/bin/spark-submit /opt/spark-apps/bdp_project.py ```

After these steps the script should start and only finish after the model has been trained and evaluated.

---

## 📊 Results

| Model                | Accuracy | Recall | Time* |
|----------------------|----------|----------|--------|
| Colab Version        | 82 % | 58 % | 20.55 | 
| Local Version        | 91 % | 85 % | 14.7 |
| Original Model       | 90.2 % | 93.9 % | 63.2 |

*The time measurement is uncertain because the paper measures time in units of time and I measured in seconds. So those results can't really be compared.


🔎 **Insights**:  
While it was possible to achieve great results with the Local implementation, close to 90% of accuracy and 89% on Recall, the Colab version suffered a lot form its limited resources on the free version of the tool. Because of that the dataset had to be substantially reduced so that a lockup wouldn't happen during the Spark processing. With this disadvantadge, the Colab version could only achieve 82% of Accuracy and 58% of Recall. 

Comparing the original experiment with the best attempt, the local implementation, we could achieve similar results, but since changes were made to the implementation described in the paper, we can't fully compare the results achieved.

More than faithfully reproducing the experiments and result, the major purpose of this project was to practice using some of the big data tools available nowadays, which was achieved.
