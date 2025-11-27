# Base: Python 3.11 (Debian Bullseye, não slim)
FROM python:3.11-bullseye

# Variáveis de ambiente Spark/Hadoop
ENV SPARK_VERSION=3.5.1
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    curl \
    openjdk-11-jdk-headless \
    git \
    unzip \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Baixar Spark pré-compilado com Hadoop
RUN curl -L https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    | tar -xz -C /opt && mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark

# Instalar pacotes Python necessários
RUN pip install --no-cache-dir numpy pandas scikit-learn tensorflow==2.19.0 joblib tqdm

# Criar diretório para scripts e dados
WORKDIR /opt/spark-apps

# Manter container rodando com Bash (para depuração) ou Spark master
CMD ["bash"]
