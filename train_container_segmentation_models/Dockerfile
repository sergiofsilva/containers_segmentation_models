# Imagem base sem GPU (MacOS não suporta GPU no Docker)
FROM tensorflow/tensorflow:2.10.0

# Instala dependências básicas e ferramentas para swap
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    linux-image-extra-virtual \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Diretório de trabalho
WORKDIR /app

# Copia requirements e instala dependências
COPY requirements_docker.txt /app/
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copia a aplicação
COPY . .

# Força o TensorFlow a usar CPU
ENV CUDA_VISIBLE_DEVICES='-1'

# Comando para executar o treinamento
CMD ["python", "train.py"]