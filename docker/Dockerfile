FROM ubuntu:16.04

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3.5 \
         nginx \
		 libgcc-5-dev \
         ca-certificates \
	 libevent-dev \
	 python-all-dev \
	 python3-dev \
    && rm -rf /var/lib/apt/lists 
RUN wget https://bootstrap.pypa.io/3.3/get-pip.py && python3.5 get-pip.py
# gevent requires pip 18 or above https://www.gevent.org/development/installing_from_source.html
RUN pip install --upgrade pip
RUN pip3 install numpy==1.16.2 scipy==1.2.1 scikit-learn==0.20.2 pandas flask statsmodels
RUN pip3 install gevent==1.5.0
RUN pip3 install gunicorn && \
        (cd /usr/local/lib/python3.5/dist-packages/scipy/.libs; rm *; ln ../../numpy/.libs/* .) && \
        rm -rf /root/.cache

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY anomaly_model /opt/program
WORKDIR /opt/program
RUN chmod u+x serve train predictor.py GESD.py

