FROM pytorch/pytorch
WORKDIR /home

COPY requirements.txt /home/requirements.txt
RUN pip3 install /home/requirements.txt
COPY download_dataset.py /home/download_dataset.py
RUN mkdir /home/data
RUN python3 /home/download_dataset.txt --dir=/home/data --subset=train
RUN python3 /home/download_dataset.txt --dir=/home/data --subset=val

RUN git clone https://github.com/iolkhovsky/detr.git

ENTRYPOINT ["/bin/bash"]
