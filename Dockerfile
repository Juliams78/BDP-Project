# Base: Python 3.11 (Debian Bullseye, not slim)
FROM python:3.11-bullseye

# Ambient variables Spark/Hadoop
ENV SPARK_VERSION=3.5.1
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH

# Installing system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    openjdk-11-jdk-headless \
    git \
    unzip \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download pre-compiled Spark-Hadoop
RUN curl -L https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    | tar -xz -C /opt && mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark

# Installing necessary python packages
RUN pip install --no-cache-dir numpy pandas matplotlib seaborn scikit-learn tensorflow==2.19.0 joblib tqdm

# Creating script and data directory
WORKDIR /opt/spark-apps

# Keep container running with Bash (for debugging) or Spark Master
CMD ["bash"]
