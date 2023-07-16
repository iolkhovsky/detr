FROM pytorch/pytorch
WORKDIR /home

COPY requirements.txt /home/requirements.txt
RUN pip3 install -r /home/requirements.txt
COPY download_dataset.py /home/download_dataset.py
RUN mkdir /home/data
RUN mkdir /home/temp
RUN python3 /home/download_dataset.py --dir=/home/data --subset=train
RUN python3 /home/download_dataset.py --dir=/home/data --subset=val

RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/iolkhovsky/detr.git

ENTRYPOINT ["/bin/bash"]
